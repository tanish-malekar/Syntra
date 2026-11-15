import Foundation

/// What the local agent reports on `GET /status` (agent/syntra_agent/client.py `Agent.status`).
struct AgentStatus: Decodable, Equatable {
    struct GPU: Decodable, Equatable, Identifiable {
        let index: Int
        let name: String
        let device: String
        let platformId: Int?
        let totalMemoryMb: Int
        let status: String          // idle | leased | busy_external | offline
        let busyDetail: String
        let workers: Int
        var id: Int { index }
    }
    let state: String               // starting | connected | reconnecting | paused | revoked | stopped
    let paused: Bool
    let connected: Bool
    let lidClosed: Bool
    let server: String
    let providerName: String
    let lastHeartbeatOk: Double?
    let lastError: String
    let draining: Bool
    let gpus: [GPU]
}

/// `GET /agent/usage` on the platform (backend/app/api/provider_app.py).
struct Usage: Decodable, Equatable {
    struct Current: Decodable, Equatable {
        let jobKind: String
        let since: String?
    }
    struct Day: Decodable, Equatable {
        let day: String
        let gpuSeconds: Double
        let amount: Double
    }
    struct GPU: Decodable, Equatable, Identifiable {
        let gpuId: Int
        let name: String
        let gpuClass: String
        let pricePerHour: Double
        let status: String
        let inUse: Bool
        let current: Current?
        let gpuSecondsTotal: Double
        let gpuSecondsRunning: Double
        let earnedSettled: Double
        let earnedRunningEstimate: Double
        let earnedTotal: Double
        let gpuSecondsToday: Double
        let earnedToday: Double
        let jobsServed: Int
        let lastDays: [Day]
        var id: Int { gpuId }
    }
    struct Totals: Decodable, Equatable {
        let gpuSecondsTotal: Double
        let gpuSecondsRunning: Double
        let earnedSettled: Double
        let earnedRunningEstimate: Double
        let earnedTotal: Double
        let gpuSecondsToday: Double
        let earnedToday: Double
        let jobsServed: Double
        let inUse: Bool
    }
    let providerName: String
    let machine: String
    let gpus: [GPU]
    let machineTotal: Totals
    let accountEarnedSettled: Double
}

struct PairResponse: Decodable {
    let token: String
    let tokenId: Int
    let providerId: Int
    let providerName: String
    let email: String
}

/// The single state the menu bar shows. Derived from the agent process, its status and the platform's usage.
enum ShareState: Equatable {
    case signedOut
    case installing(String)          // current install step
    case notSharing                  // agent not running (user pressed Stop / never started)
    case starting
    case sharedIdle
    case inUse
    case paused(draining: Bool)
    case reconnecting(String)
    case busyWithOwnerApps
    case error(String)

    var title: String {
        switch self {
        case .signedOut: return "Signed out"
        case .installing(let step): return "Installing: \(step)"
        case .notSharing: return "Not shared"
        case .starting: return "Connecting…"
        case .sharedIdle: return "Shared: waiting for work"
        case .inUse: return "Shared: in use"
        case .paused(let draining): return draining ? "Pausing: finishing the current job" : "Paused"
        case .reconnecting: return "Disconnected: reconnecting…"
        case .busyWithOwnerApps: return "Busy with your apps (not shared)"
        case .error(let message): return "Error: \(message)"
        }
    }

    /// SF Symbol for the menu bar.
    var symbol: String {
        switch self {
        case .inUse: return "bolt.circle.fill"
        case .sharedIdle: return "checkmark.circle.fill"
        case .starting, .installing: return "arrow.triangle.2.circlepath.circle"
        case .paused, .notSharing, .signedOut: return "pause.circle"
        case .busyWithOwnerApps: return "person.crop.circle.badge.clock"
        case .reconnecting, .error: return "exclamationmark.triangle.fill"
        }
    }

    static func derive(running: Bool, status: AgentStatus?) -> ShareState {
        guard running else { return .notSharing }
        guard let s = status else { return .starting }
        if s.state == "revoked" { return .error("signed out on the website; sign in again") }
        if s.paused { return .paused(draining: s.draining) }
        if s.state == "starting" { return .starting }
        if s.state == "reconnecting" || !s.connected { return .reconnecting(s.lastError) }
        if s.gpus.contains(where: { $0.status == "leased" }) { return .inUse }
        if !s.gpus.isEmpty && s.gpus.allSatisfy({ $0.status == "busy_external" }) { return .busyWithOwnerApps }
        return .sharedIdle
    }
}

enum Format {
    static func money(_ v: Double) -> String { String(format: "$%.2f", v) }

    static func duration(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        if s < 60 { return "\(s) s" }
        if s < 3600 { return "\(s / 60) min" }
        let h = s / 3600, m = (s % 3600) / 60
        return m == 0 ? "\(h) h" : "\(h) h \(m) min"
    }
}

extension JSONDecoder {
    static var snake: JSONDecoder {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }
}
