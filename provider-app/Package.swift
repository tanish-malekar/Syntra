// swift-tools-version:5.9
// BloomBee Provider: a macOS menu bar app that shares this Mac's GPU with the BloomBee pool with one click.
import PackageDescription

let package = Package(
    name: "BloomBeeProvider",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "BloomBeeProvider", path: "Sources/BloomBeeProvider"),
        .testTarget(name: "BloomBeeProviderTests", dependencies: ["BloomBeeProvider"], path: "Tests/BloomBeeProviderTests"),
    ]
)
