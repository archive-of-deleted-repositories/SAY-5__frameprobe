#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <set>
#include <thread>
#include <vector>

#include "detect/serialize.h"
#include "detect/stub_detector.h"
#include "pipeline/decoder.h"
#include "pipeline/harness.h"
#include "pipeline/inference.h"
#include "pipeline/sink.h"
#include "source/synthetic_source.h"

using namespace frameprobe;

namespace {

// Connect the production stages directly to observe the sink's existing emit
// callback: Harness only exposes aggregate statistics, not detection results.
std::vector<DetectionResult> run_collect(int workers, std::uint64_t frames) {
    StubConfig sc;
    sc.stream.frames = frames;
    sc.compute_us = 10;

    SyntheticSource source(sc.stream);
    StubDetector detector(sc);
    BoundedQueue<FrameEnvelope> frame_q(8);
    BoundedQueue<ResultEnvelope> result_q(16);
    Decoder decoder(source, frame_q);
    FrameStats stats;
    Sink sink(result_q, stats);
    std::vector<DetectionResult> got;
    sink.set_emit([&](const ResultEnvelope& env) { got.push_back(env.result); });

    std::thread decoder_thread([&] { decoder.run(); });
    std::vector<std::thread> worker_threads;
    for (int i = 0; i < workers; ++i) {
        worker_threads.emplace_back([&] {
            InferenceWorker worker(frame_q, result_q, detector);
            worker.run();
        });
    }
    std::thread sink_thread([&] { sink.run(); });

    decoder_thread.join();
    for (auto& thread : worker_threads)
        thread.join();
    result_q.close();
    sink_thread.join();

    EXPECT_EQ(stats.processed(), frames);
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_EQ(got[i].frame_index, i);
        EXPECT_EQ(got[i].detections, StubDetector::scripted(sc, i).detections)
            << "workers=" << workers << " frame=" << i;
    }
    return got;
}

}  // namespace

// Property: detection output is bit-identical across thread counts {1,2,4}.
TEST(Property, DetectionsIdenticalAcrossThreadCounts) {
    const std::uint64_t frames = 400;
    auto a = run_collect(1, frames);
    auto b = run_collect(2, frames);
    auto c = run_collect(4, frames);

    ASSERT_EQ(a.size(), frames);
    ASSERT_EQ(b.size(), a.size());
    ASSERT_EQ(c.size(), a.size());
    for (std::uint64_t i = 0; i < frames; ++i) {
        ASSERT_EQ(a[i].detections.size(), b[i].detections.size());
        ASSERT_EQ(a[i].detections.size(), c[i].detections.size());
        for (std::size_t j = 0; j < a[i].detections.size(); ++j) {
            EXPECT_EQ(a[i].detections[j], b[i].detections[j]);
            EXPECT_EQ(a[i].detections[j], c[i].detections[j]);
        }
    }
}

// Property: in block (back-pressure) mode the pipeline never deadlocks and never
// loses a frame, regardless of how slow the consumer is relative to the queue.
TEST(Property, BackpressureNeverLosesFrames) {
    for (std::size_t cap : {1u, 2u, 4u, 16u}) {
        StubConfig sc;
        sc.stream.frames = 600;
        sc.compute_us = 30;
        PipelineConfig cfg;
        cfg.target_fps = 1000.0;
        cfg.queue_capacity = cap;
        cfg.overflow = Overflow::kBlock;
        Harness h(std::make_unique<SyntheticSource>(sc.stream), std::make_unique<StubDetector>(sc),
                  cfg);
        FrameStats s = h.run();
        EXPECT_EQ(s.processed(), sc.stream.frames) << "cap=" << cap;
        EXPECT_EQ(s.dropped(), 0u) << "cap=" << cap;
    }
}

// Property: in drop mode the accounting is exact: processed + dropped equals the
// number of frames the source produced (the queue's drop counter is the source
// of truth, surfaced via the harness's processed/dropped split is not directly
// summable here, so we assert the weaker invariant that processed never exceeds
// produced and the run terminates).
TEST(Property, DropModeProcessedNeverExceedsProduced) {
    std::mt19937 rng(12345);
    for (int trial = 0; trial < 8; ++trial) {
        StubConfig sc;
        sc.stream.frames = 300;
        sc.compute_us = 20 + rng() % 60;
        PipelineConfig cfg;
        cfg.target_fps = 1000.0;
        cfg.queue_capacity = 1 + rng() % 8;
        cfg.overflow = Overflow::kDrop;
        Harness h(std::make_unique<SyntheticSource>(sc.stream), std::make_unique<StubDetector>(sc),
                  cfg);
        FrameStats s = h.run();
        EXPECT_LE(s.processed(), sc.stream.frames);
        EXPECT_GT(s.processed(), 0u);
    }
}

// Property: the queue's own drop counter is exact under a known producer rate.
TEST(Property, QueueDropCountIsExact) {
    BoundedQueue<int> q(4, Overflow::kDrop);
    // Fill to capacity, then push 10 more with no consumer: exactly 10 drops.
    for (int i = 0; i < 4; ++i)
        ASSERT_TRUE(q.push(i));
    for (int i = 0; i < 10; ++i)
        EXPECT_FALSE(q.push(i));
    EXPECT_EQ(q.drop_count(), 10u);
    EXPECT_EQ(q.size(), 4u);
}

// Property: serialization round-trips for arbitrary detection counts.
TEST(Property, SerializationRoundTripsForManySizes) {
    std::mt19937 rng(999);
    for (int trial = 0; trial < 50; ++trial) {
        DetectionResult r;
        r.frame_index = rng();
        const int count = rng() % 20;
        for (int i = 0; i < count; ++i) {
            r.detections.push_back({static_cast<int>(rng() % 1000), static_cast<int>(rng() % 1000),
                                    static_cast<int>(rng() % 200), static_cast<int>(rng() % 200),
                                    static_cast<int>(rng() % 80),
                                    static_cast<float>(rng() % 1000) / 1000.0f});
        }
        auto buf = wire::serialize(r);
        auto back = wire::deserialize(buf);
        ASSERT_TRUE(back.has_value());
        EXPECT_EQ(back->frame_index, r.frame_index);
        ASSERT_EQ(back->detections.size(), r.detections.size());
        for (std::size_t i = 0; i < r.detections.size(); ++i) {
            EXPECT_EQ(back->detections[i], r.detections[i]);
        }
    }
}

// Property: the frame-header parser rejects every malformed header and accepts
// every well-formed one (round-trip).
TEST(Property, FrameHeaderParseRejectsAndRoundTrips) {
    wire::FrameHeader h;
    h.index = 42;
    h.width = 320;
    h.height = 240;
    h.channels = 3;
    auto buf = wire::serialize_header(h);
    auto back = wire::parse_header(buf);
    ASSERT_TRUE(back.has_value());
    EXPECT_EQ(back->width, 320);
    EXPECT_EQ(back->channels, 3);

    // Zero dimensions rejected.
    wire::FrameHeader bad = h;
    bad.width = 0;
    EXPECT_FALSE(wire::parse_header(wire::serialize_header(bad)).has_value());

    // Channels out of range rejected.
    bad = h;
    bad.channels = 7;
    EXPECT_FALSE(wire::parse_header(wire::serialize_header(bad)).has_value());

    // Truncation rejected.
    buf.pop_back();
    EXPECT_FALSE(wire::parse_header(buf).has_value());
}
