#!/usr/bin/env bash
# Build BloomBeeProvider.app (window + Dock icon + menu bar icon) into provider-app/dist/.
#   ./build.sh            release build
#   ./build.sh --open     build and launch
set -euo pipefail
cd "$(dirname "$0")"
swift build -c release
APP=dist/BloomBeeProvider.app
rm -rf "$APP" && mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$(swift build -c release --show-bin-path)/BloomBeeProvider" "$APP/Contents/MacOS/BloomBeeProvider"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>ai.bloombee.provider</string>
  <key>CFBundleName</key><string>BloomBee</string>
  <key>CFBundleDisplayName</key><string>BloomBee Provider</string>
  <key>CFBundleExecutable</key><string>BloomBeeProvider</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSUIElement</key><false/>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/><key>NSAllowsArbitraryLoads</key><true/></dict>
</dict></plist>
PLIST
# Sign with a stable identity when there is one (an ad-hoc signature changes on every build, and macOS then asks for the
# login password before the app may read its own Keychain item). Override with BLOOMBEE_SIGN_IDENTITY.
IDENTITY="${BLOOMBEE_SIGN_IDENTITY:-$(security find-identity -v -p codesigning 2>/dev/null | awk -F'"' '/Apple Development|Developer ID Application/ {print $2; exit}')}"
codesign --force --sign "${IDENTITY:--}" "$APP" >/dev/null 2>&1 || codesign --force --sign - "$APP" >/dev/null 2>&1 || true
echo "Signed with: ${IDENTITY:-ad-hoc}"
echo "Built $APP"
[ "${1:-}" = "--open" ] && open "$APP"
exit 0
