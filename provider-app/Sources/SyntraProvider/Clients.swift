import Foundation
import IOKit
import Security

enum ClientError: LocalizedError {
    case http(Int, String)
    case notRunning
    case message(String)

    var errorDescription: String? {
        switch self {
        case .http(let code, let body):
            if let data = body.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let detail = obj["detail"] as? String ?? obj["error"] as? String {
                return detail
            }
            return "HTTP \(code)"
        case .notRunning: return "The agent is not running"
        case .message(let m): return m
        }
    }
}

private func send(_ request: URLRequest) async throws -> Data {
    let (data, response) = try await URLSession.shared.data(for: request)
    let code = (response as? HTTPURLResponse)?.statusCode ?? 0
    guard (200..<300).contains(code) else { throw ClientError.http(code, String(decoding: data, as: UTF8.self)) }
    return data
}

/// The Syntra platform (backend): pairing, usage, sign-out.
struct PlatformClient {
    let server: URL

    func health() async throws {
        var r = URLRequest(url: server.appendingPathComponent("health"))
        r.timeoutInterval = 8
        _ = try await send(r)
    }

    func pair(email: String, password: String, machineId: String, machineName: String) async throws -> PairResponse {
        var r = URLRequest(url: server.appendingPathComponent("app/pair"))
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.httpBody = try JSONSerialization.data(withJSONObject: [
            "email": email, "password": password, "machine_id": machineId, "machine_name": machineName,
        ])
        r.timeoutInterval = 15
        return try JSONDecoder.snake.decode(PairResponse.self, from: try await send(r))
    }

    func usage(token: String) async throws -> Usage {
        var r = URLRequest(url: server.appendingPathComponent("agent/usage"))
        r.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        r.timeoutInterval = 10
        return try JSONDecoder.snake.decode(Usage.self, from: try await send(r))
    }

    func unpair(token: String) async throws {
        var r = URLRequest(url: server.appendingPathComponent("agent/unpair"))
        r.httpMethod = "POST"
        r.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        r.timeoutInterval = 10
        _ = try await send(r)
    }
}

/// The agent's local control API (agent/syntra_agent/control.py). Port and secret come from ~/.syntra/agent.json, so the
/// app also attaches to an agent that was started from a terminal.
struct ControlClient {
    let stateDir: URL

    struct Info: Decodable { let port: Int; let secret: String; let pid: Int32 }

    func info() -> Info? {
        guard let data = try? Data(contentsOf: stateDir.appendingPathComponent("agent.json")),
              let info = try? JSONDecoder().decode(Info.self, from: data), kill(info.pid, 0) == 0 else { return nil }
        return info
    }

    private func request(_ method: String, _ path: String, body: [String: Any]? = nil) async throws -> AgentStatus {
        guard let info = info() else { throw ClientError.notRunning }
        var r = URLRequest(url: URL(string: "http://127.0.0.1:\(info.port)\(path)")!)
        r.httpMethod = method
        r.setValue(info.secret, forHTTPHeaderField: "X-Syntra-Secret")
        r.timeoutInterval = 5
        if let body {
            r.setValue("application/json", forHTTPHeaderField: "Content-Type")
            r.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        return try JSONDecoder.snake.decode(AgentStatus.self, from: try await send(r))
    }

    func status() async throws -> AgentStatus { try await request("GET", "/status") }
    func pause(now: Bool) async throws -> AgentStatus { try await request("POST", "/pause", body: ["mode": now ? "now" : "drain"]) }
    func resume() async throws -> AgentStatus { try await request("POST", "/resume") }
    func quit() async throws -> AgentStatus { try await request("POST", "/quit") }
}

/// The agent token lives in the login Keychain, never in a file.
enum Keychain {
    static let service = "ai.syntra.provider"

    static func save(_ value: String, account: String) throws {
        delete(account: account)
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                    kSecAttrAccount as String: account, kSecValueData as String: Data(value.utf8),
                                    kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly]
        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else { throw ClientError.message("Keychain error \(status)") }
    }

    static func load(account: String) -> String? {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                    kSecAttrAccount as String: account, kSecReturnData as String: true,
                                    kSecMatchLimit as String: kSecMatchLimitOne]
        var out: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &out) == errSecSuccess, let data = out as? Data else { return nil }
        return String(decoding: data, as: UTF8.self)
    }

    static func delete(account: String) {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                    kSecAttrAccount as String: account]
        SecItemDelete(query as CFDictionary)
    }
}

enum Machine {
    /// Stable per Mac: the hardware UUID (IOPlatformUUID). Re-pairing the same Mac replaces its old token.
    static var id: String {
        let entry = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("IOPlatformExpertDevice"))
        defer { IOObjectRelease(entry) }
        if let uuid = IORegistryEntryCreateCFProperty(entry, "IOPlatformUUID" as CFString, kCFAllocatorDefault, 0)?
            .takeRetainedValue() as? String {
            return uuid
        }
        return Host.current().name ?? "unknown-mac"
    }

    static var name: String { Host.current().localizedName ?? "Mac" }
}
