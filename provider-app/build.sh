#!/usr/bin/env bash
# Build SyntraProvider.app (menu bar only, no Dock icon) into provider-app/dist/.
#   ./build.sh            release build
#   ./build.sh --open     build and launch
set -euo pipefail
cd "$(dirname "$0")"
swift build -c release
APP=dist/SyntraProvider.app
rm -rf "$APP" && mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$(swift build -c release --show-bin-path)/SyntraProvider" "$APP/Contents/MacOS/SyntraProvider"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>ai.syntra.provider</string>
  <key>CFBundleName</key><string>Syntra</string>
  <key>CFBundleDisplayName</key><string>Syntra Provider</string>
  <key>CFBundleExecutable</key><string>SyntraProvider</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSUIElement</key><true/>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/><key>NSAllowsArbitraryLoads</key><true/></dict>
</dict></plist>
PLIST
codesign --force --sign - "$APP" >/dev/null 2>&1 || true   # ad-hoc signature so Keychain and login items work locally
echo "Built $APP"
[ "${1:-}" = "--open" ] && open "$APP"
exit 0
