import XCTest
@testable import SyntraProvider

final class ShareStateTests: XCTestCase {
    func status(_ state: String = "connected", paused: Bool = false, connected: Bool = true, draining: Bool = false,
                gpus: [String] = ["idle"]) -> AgentStatus {
        let json: [String: Any] = [
            "state": state, "paused": paused, "connected": connected, "lid_closed": false, "server": "http://x",
            "provider_name": "P", "last_heartbeat_ok": 1.0, "last_error": "", "draining": draining,
            "gpus": gpus.enumerated().map { i, s in ["index": i, "name": "M4", "device": "mps", "platform_id": i,
                                                     "total_memory_mb": 16000, "status": s, "busy_detail": "", "workers": 0] },
        ]
        return try! JSONDecoder.snake.decode(AgentStatus.self, from: JSONSerialization.data(withJSONObject: json))
    }

    func testDerive() {
        XCTAssertEqual(ShareState.derive(running: false, status: nil), .notSharing)
        XCTAssertEqual(ShareState.derive(running: true, status: nil), .starting)
        XCTAssertEqual(ShareState.derive(running: true, status: status()), .sharedIdle)
        XCTAssertEqual(ShareState.derive(running: true, status: status(gpus: ["leased", "idle"])), .inUse)
        XCTAssertEqual(ShareState.derive(running: true, status: status(gpus: ["busy_external"])), .busyWithOwnerApps)
        XCTAssertEqual(ShareState.derive(running: true, status: status(paused: true, draining: true, gpus: ["leased"])), .paused(draining: true))
        XCTAssertEqual(ShareState.derive(running: true, status: status("reconnecting", connected: false)), .reconnecting(""))
        if case .error = ShareState.derive(running: true, status: status("revoked")) {} else { XCTFail("revoked must be an error") }
    }

    func testFormat() {
        XCTAssertEqual(Format.duration(42), "42 s")
        XCTAssertEqual(Format.duration(125), "2 min")
        XCTAssertEqual(Format.duration(3600), "1 h")
        XCTAssertEqual(Format.duration(5400), "1 h 30 min")
        XCTAssertEqual(Format.money(1.234), "$1.23")
    }

    func testUsageDecodes() throws {
        let json = """
        {"provider_name":"P","machine":"Mac","account_earned_settled":1.5,
         "machine_total":{"gpu_seconds_total":10,"gpu_seconds_running":5,"earned_settled":1,"earned_running_estimate":0.5,
                          "earned_total":1.5,"gpu_seconds_today":10,"earned_today":1.5,"jobs_served":2,"in_use":true},
         "gpus":[{"gpu_id":1,"name":"M4","device":"mps","gpu_class":"Apple M-series","price_per_hour":0.1,"status":"leased",
                  "in_use":true,"current":{"job_kind":"train","since":"2026-11-14T00:00:00Z","layers":[0,22]},
                  "gpu_seconds_total":10,"gpu_seconds_running":5,"earned_settled":1,"earned_running_estimate":0.5,
                  "earned_total":1.5,"gpu_seconds_today":10,"earned_today":1.5,"jobs_served":2,
                  "last_days":[{"day":"2026-11-14","gpu_seconds":5,"amount":1}]}]}
        """
        let u = try JSONDecoder.snake.decode(Usage.self, from: Data(json.utf8))
        XCTAssertEqual(u.gpus.first?.current?.jobKind, "train")
        XCTAssertTrue(u.machineTotal.inUse)
    }
}
