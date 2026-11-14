// swift-tools-version:5.9
// Syntra Provider: a macOS menu bar app that shares this Mac's GPU with the Syntra pool with one click.
import PackageDescription

let package = Package(
    name: "SyntraProvider",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "SyntraProvider", path: "Sources/SyntraProvider"),
        .testTarget(name: "SyntraProviderTests", dependencies: ["SyntraProvider"], path: "Tests/SyntraProviderTests"),
    ]
)
