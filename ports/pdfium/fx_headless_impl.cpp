// Headless platform implementation for static PDFium builds.
// Maintained by Talu project.
//
// WHY HEADLESS?
// Platform-specific code (CoreGraphics on macOS, system font paths on Linux)
// creates runtime dependencies that prevent fully static builds. This stub
// provides a platform implementation with no external dependencies.
//
// WHAT THIS DISABLES:
// - System font enumeration (searching /usr/share/fonts, macOS font dirs, etc.)
// - Native text rendering (CoreGraphics subpixel antialiasing on macOS)
// - System font substitution (finding "Arial" when PDF references but doesn't embed it)
//
// WHY THIS DOESN'T AFFECT TALU:
// Talu uses PDFium for PDF content extraction (text, images, metadata), not
// rendering. Extraction uses the PDF's internal font tables to decode text -
// no font rendering or system fonts needed. The disabled features only matter
// when rendering PDFs to images/display.

#include <memory>

#include "build/build_config.h"
#include "core/fxcrt/bytestring.h"
#include "core/fxcrt/fx_codepage.h"
#include "core/fxcrt/span.h"
#include "core/fxge/cfx_fontmapper.h"
#include "core/fxge/cfx_gemodule.h"
#include "core/fxge/systemfontinfo_iface.h"

namespace {

// Stub SystemFontInfo that doesn't enumerate system fonts.
// PDFium will fall back to embedded fonts in the PDF itself.
class CHeadlessFontInfo final : public SystemFontInfoIface {
 public:
  CHeadlessFontInfo() = default;
  ~CHeadlessFontInfo() override = default;

  void EnumFontList(CFX_FontMapper* pMapper) override {}
  void* MapFont(int weight,
                bool bItalic,
                FX_Charset charset,
                int pitch_family,
                const ByteString& face) override {
    return nullptr;
  }
  void* GetFont(const ByteString& face) override { return nullptr; }
  size_t GetFontData(void* hFont,
                     uint32_t table,
                     pdfium::span<uint8_t> buffer) override {
    return 0;
  }
  bool GetFaceName(void* hFont, ByteString* name) override { return false; }
  bool GetFontCharset(void* hFont, FX_Charset* charset) override {
    return false;
  }
  void DeleteFont(void* hFont) override {}
};

}  // namespace

// Headless platform - no system font rendering, no framework dependencies.
// When PDFIUM_HEADLESS is defined, IS_APPLE evaluates to 0, so we don't need
// the CreatePlatformFont method.
class CHeadlessPlatform : public CFX_GEModule::PlatformIface {
 public:
  CHeadlessPlatform() = default;
  ~CHeadlessPlatform() override = default;

  void Init() override {}

  std::unique_ptr<SystemFontInfoIface> CreateDefaultSystemFontInfo() override {
    return std::make_unique<CHeadlessFontInfo>();
  }
};

// static
std::unique_ptr<CFX_GEModule::PlatformIface>
CFX_GEModule::PlatformIface::Create() {
  return std::make_unique<CHeadlessPlatform>();
}
