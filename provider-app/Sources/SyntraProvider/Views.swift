import SwiftUI

@main
struct SyntraProviderApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        MenuBarExtra {
            PanelView().environmentObject(model)
        } label: {
            Image(systemName: model.share.symbol)
        }
        .menuBarExtraStyle(.window)
    }
}

struct PanelView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Syntra").font(.headline)
                Spacer()
                Text(model.share.title).font(.subheadline).foregroundStyle(color)
            }
            if model.signedIn { SharingView() } else { SignInView() }
            if let flash = model.flash {
                Text(flash).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Divider()
            HStack {
                if model.signedIn {
                    Menu("More") {
                        Toggle("Pause when on battery", isOn: $model.pauseOnBattery)
                        Toggle("Start at login", isOn: Binding(get: { model.startsAtLogin }, set: { model.setStartsAtLogin($0) }))
                        Button("Open agent log") { model.openLog() }
                        Button("Sign out of this Mac") { Task { await model.signOut() } }
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                }
                Spacer()
                Button("Quit") { model.quitApp() }
            }
        }
        .padding(14)
        .frame(width: 340)
    }

    var color: Color {
        switch model.share {
        case .inUse: return .blue
        case .sharedIdle: return .green
        case .reconnecting, .error: return .red
        case .busyWithOwnerApps: return .orange
        default: return .secondary
        }
    }
}

struct SignInView: View {
    @EnvironmentObject var model: AppModel
    @State private var password = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Sign in to share this Mac's GPU and earn from it.").font(.callout)
            TextField("Server", text: $model.serverText).textFieldStyle(.roundedBorder)
            TextField("Email", text: $model.email).textFieldStyle(.roundedBorder).textContentType(.username)
            SecureField("Password", text: $password).textFieldStyle(.roundedBorder)
                .onSubmit { Task { await model.signIn(password: password) } }
            Toggle("Works on any network (relay)", isOn: $model.useRelay).font(.caption)
            Button {
                Task { await model.signIn(password: password); password = "" }
            } label: {
                Text(model.busy ? "Signing in…" : "Sign in").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(model.busy || model.email.isEmpty || password.isEmpty)
        }
    }
}

struct SharingView: View {
    @EnvironmentObject var model: AppModel
    @State private var confirmPause = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            mainButton
            if case .reconnecting(let why) = model.share, !why.isEmpty {
                Text(why).font(.caption).foregroundStyle(.red).lineLimit(3)
            }
            if case .error(let why) = model.share {
                Text(why).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
                Button("Open agent log") { model.openLog() }.font(.caption)
            }
            if let status = model.status, !status.gpus.isEmpty {
                ForEach(status.gpus) { gpu in
                    HStack {
                        Circle().fill(dot(gpu.status)).frame(width: 8, height: 8)
                        Text(gpu.name).font(.callout)
                        Spacer()
                        Text(label(gpu)).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            UsageView()
        }
        .confirmationDialog("A job is using this GPU", isPresented: $confirmPause) {
            Button("Finish the current job, then pause") { Task { await model.pause(now: false) } }
            Button("Pause now (the job moves to another GPU)", role: .destructive) { Task { await model.pause(now: true) } }
        }
    }

    @ViewBuilder var mainButton: some View {
        switch model.share {
        case .notSharing, .error:
            Button { Task { await model.startSharing() } } label: {
                Label("Share my GPU", systemImage: "play.fill").frame(maxWidth: .infinity)
            }.buttonStyle(.borderedProminent).controlSize(.large).disabled(model.busy)
        case .paused:
            HStack {
                Button { Task { await model.resume() } } label: { Label("Resume sharing", systemImage: "play.fill").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent)
                Button("Stop") { Task { await model.stopSharing() } }
            }.controlSize(.large)
        case .installing, .starting:
            HStack { ProgressView().controlSize(.small); Text(model.share.title).font(.callout) }
        default:
            HStack {
                Button {
                    if model.share == .inUse { confirmPause = true } else { Task { await model.pause(now: false) } }
                } label: { Label("Pause", systemImage: "pause.fill").frame(maxWidth: .infinity) }
                Button("Stop") { Task { await model.stopSharing() } }
            }.controlSize(.large)
        }
    }

    func dot(_ status: String) -> Color {
        switch status {
        case "leased": return .blue
        case "idle": return .green
        case "busy_external": return .orange
        default: return .gray
        }
    }

    func label(_ gpu: AgentStatus.GPU) -> String {
        switch gpu.status {
        case "leased": return "in use"
        case "idle": return "waiting for work"
        case "busy_external": return "busy with your apps"
        default: return model.status?.lidClosed == true ? "lid closed" : "offline"
        }
    }
}

struct UsageView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        if let u = model.usage {
            let t = u.machineTotal
            VStack(alignment: .leading, spacing: 6) {
                Divider()
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                    GridRow { Text("").gridColumnAlignment(.leading); Text("Today").bold(); Text("All time").bold() }
                    GridRow { Text("GPU time used"); Text(Format.duration(t.gpuSecondsToday)); Text(Format.duration(t.gpuSecondsTotal)) }
                    GridRow { Text("Earned"); Text(Format.money(t.earnedToday)); Text(Format.money(t.earnedTotal)) }
                }
                .font(.callout)
                if t.earnedRunningEstimate > 0 {
                    Text("Includes ≈\(Format.money(t.earnedRunningEstimate)) for the job running now (settled when it ends).")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                ForEach(u.gpus.filter { $0.inUse }) { g in
                    Text("\(g.name): running a \(g.current?.jobKind ?? "job") job for \(Format.duration(g.gpuSecondsRunning)) at \(Format.money(g.pricePerHour))/h")
                        .font(.caption).foregroundStyle(.blue)
                }
                Text("\(Int(t.jobsServed)) jobs served · account total \(Format.money(u.accountEarnedSettled)) settled")
                    .font(.caption).foregroundStyle(.secondary)
            }
        } else if let err = model.lastUsageError {
            Text("Usage unavailable: \(err)").font(.caption).foregroundStyle(.secondary)
        }
    }
}
