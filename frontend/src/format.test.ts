import { describe, expect, it } from "vitest";
import type { StoredEvent } from "./api";
import { ago, describeEvent, duration, layersLabel, lossSeries, mb, money, pct, stageStates } from "./format";

const ev = (type: string, payload: Record<string, any>, id = 1): StoredEvent => ({ id, type, ts: "2026-01-01T00:00:00Z", job_id: 1, gpu_id: null, payload });

describe("formatting", () => {
  it("formats memory, money, durations and percentages", () => {
    expect(mb(512)).toBe("512 MB");
    expect(mb(6144)).toBe("6 GB");
    expect(mb(24576)).toBe("24 GB");
    expect(mb(1536)).toBe("1.5 GB");
    expect(money(0.2)).toBe("$0.2000");
    expect(money(12.5)).toBe("$12.50");
    expect(duration(2.83)).toBe("2.8 s");
    expect(duration(45)).toBe("45 s");
    expect(duration(125)).toBe("2 min 5 s");
    expect(duration(3600)).toBe("1 h");
    expect(duration(38700)).toBe("10 h 45 min");
    expect(pct(47, 48)).toBe("98%");
    expect(pct(0, 0)).toBe("–");
    expect(layersLabel(11, 22)).toBe("layers 11–21");
    expect(layersLabel(3, 4)).toBe("layer 3");
  });
  it("describes how long ago something happened", () => {
    const now = new Date("2026-01-01T00:10:00Z").getTime();
    expect(ago(null, now)).toBe("never");
    expect(ago("2026-01-01T00:09:58Z", now)).toBe("just now");
    expect(ago("2026-01-01T00:09:30Z", now)).toBe("30 s ago");
    expect(ago("2026-01-01T00:05:00Z", now)).toBe("5 min ago");
  });
});

describe("run timeline", () => {
  it("explains a failure and the replacement GPU in plain language", () => {
    const down = describeEvent(ev("worker.down", { reason: "killed_by_user", layer_start: 11, layer_end: 22 }))!;
    expect(down.tone).toBe("bad");
    expect(down.title).toContain("you stopped it");
    expect(down.detail).toBe("layers 11–21");
    const start = describeEvent(ev("worker.replacement_starting", {
      layer_start: 11, layer_end: 22, gpu_name: "Apple M4 Pro (sim 3)", gpu_device: "mps", gpu_class: "Apple M-series", gpu_memory_mb: 6144,
    }))!;
    expect(start.title).toBe("New GPU found: Apple M4 Pro (sim 3)");
    expect(start.detail).toBe("Apple M-series · MPS · 6 GB · taking over layers 11–21");
    const online = describeEvent(ev("worker.replacement_online", { layer_start: 11, layer_end: 22, peer_id: "x", seconds_since_down: 2.83 }))!;
    expect(online.detail).toContain("2.8 s after the failure");
  });
  it("does not list training steps or pool heartbeats as timeline entries", () => {
    expect(describeEvent(ev("train.step", { step: 1 }))).toBeNull();
    expect(describeEvent(ev("gpu.heartbeat", {}))).toBeNull();
  });
});

describe("loss series and stage state", () => {
  it("orders steps, keeps the latest value per step", () => {
    const series = lossSeries([
      ev("train.step", { step: 2, loss: 1.9 }, 2), ev("train.step", { step: 1, loss: 2.1 }, 1), ev("train.step", { step: 2, loss: 1.8 }, 3),
      ev("worker.online", { layer_start: 0 }, 4),
    ]);
    expect(series).toEqual([{ step: 1, loss: 2.1 }, { step: 2, loss: 1.8 }]);
  });
  it("tracks a stage through starting, online, down, replacing and online again", () => {
    const events = [
      ev("lease.created", { layer_start: 11, layer_end: 22 }, 1), ev("worker.online", { layer_start: 11, layer_end: 22 }, 2),
    ];
    expect(stageStates(events).get(11)).toBe("online");
    events.push(ev("worker.down", { layer_start: 11, layer_end: 22, reason: "exited" }, 3));
    expect(stageStates(events).get(11)).toBe("down");
    events.push(ev("worker.replacement_starting", { layer_start: 11, layer_end: 22 }, 4));
    expect(stageStates(events).get(11)).toBe("replacing");
    events.push(ev("worker.replacement_online", { layer_start: 11, layer_end: 22 }, 5));
    expect(stageStates(events).get(11)).toBe("online");
  });
});
