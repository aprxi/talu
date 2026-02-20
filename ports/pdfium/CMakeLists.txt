# PDFium CMake build — maintained by Talu project.
# Reference: OlexiyKhokhlov/PDFium CMake fork.
# This file is copied into the PDFium source tree during `make deps`.
cmake_minimum_required(VERSION 3.16)
project(pdfium VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

if(NOT WIN32)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>)
    add_compile_options(-fvisibility=hidden)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type>)
endif()

# ---------------------------------------------------------------------------
# Dependencies — all passed explicitly (no find_package for portability)
# ---------------------------------------------------------------------------

# ICU (only headers needed; actual symbols provided by icu_shim.c at link time)
if(NOT ICU_INCLUDE_DIR)
    message(FATAL_ERROR "ICU_INCLUDE_DIR must be set (path containing unicode/uchar.h)")
endif()
set(ICU_INCLUDE_DIRS ${ICU_INCLUDE_DIR})
add_definitions(-DUSE_SYSTEM_ICUUC)

# Freetype
if(NOT FREETYPE_INCLUDE_DIR OR NOT FREETYPE_LIBRARY)
    message(FATAL_ERROR "FREETYPE_INCLUDE_DIR and FREETYPE_LIBRARY must be set")
endif()
add_library(freetype_imported STATIC IMPORTED)
set_target_properties(freetype_imported PROPERTIES
    IMPORTED_LOCATION "${FREETYPE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FREETYPE_INCLUDE_DIR}"
)
add_definitions(-DUSE_SYSTEM_FREETYPE -DDEFINE_PS_TABLES_DATA)

# JPEG
if(NOT JPEG_INCLUDE_DIR)
    message(FATAL_ERROR "JPEG_INCLUDE_DIR must be set")
endif()
add_definitions(-DUSE_SYSTEM_JPEG)

# ZLIB (headers only; miniz provides symbols at link time via zlib compat macros)
if(NOT ZLIB_INCLUDE_DIR)
    message(FATAL_ERROR "ZLIB_INCLUDE_DIR must be set (path containing zlib.h)")
endif()
set(ZLIB_INCLUDE_DIRS ${ZLIB_INCLUDE_DIR})
add_definitions(-DUSE_SYSTEM_ZLIB)

# ---------------------------------------------------------------------------
# third_party/agg23
# ---------------------------------------------------------------------------
add_library(agg STATIC
    third_party/agg23/agg_curves.cpp
    third_party/agg23/agg_path_storage.cpp
    third_party/agg23/agg_rasterizer_scanline_aa.cpp
    third_party/agg23/agg_vcgen_dash.cpp
    third_party/agg23/agg_vcgen_stroke.cpp
)
target_include_directories(agg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------------------
# third_party/lcms (bundled, not system)
# ---------------------------------------------------------------------------
add_library(lcms STATIC
    third_party/lcms/src/cmsalpha.c
    third_party/lcms/src/cmscam02.c
    third_party/lcms/src/cmscgats.c
    third_party/lcms/src/cmscnvrt.c
    third_party/lcms/src/cmserr.c
    third_party/lcms/src/cmsgamma.c
    third_party/lcms/src/cmsgmt.c
    third_party/lcms/src/cmshalf.c
    third_party/lcms/src/cmsintrp.c
    third_party/lcms/src/cmsio0.c
    third_party/lcms/src/cmsio1.c
    third_party/lcms/src/cmslut.c
    third_party/lcms/src/cmsmd5.c
    third_party/lcms/src/cmsmtrx.c
    third_party/lcms/src/cmsnamed.c
    third_party/lcms/src/cmsopt.c
    third_party/lcms/src/cmspack.c
    third_party/lcms/src/cmspcs.c
    third_party/lcms/src/cmsplugin.c
    third_party/lcms/src/cmsps2.c
    third_party/lcms/src/cmssamp.c
    third_party/lcms/src/cmssm.c
    third_party/lcms/src/cmstypes.c
    third_party/lcms/src/cmsvirt.c
    third_party/lcms/src/cmswtpnt.c
    third_party/lcms/src/cmsxform.c
)
target_include_directories(lcms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------------------
# third_party/libopenjpeg (bundled, not system)
# ---------------------------------------------------------------------------
add_library(openjpeg STATIC
    third_party/libopenjpeg/bio.c
    third_party/libopenjpeg/cio.c
    third_party/libopenjpeg/dwt.c
    third_party/libopenjpeg/event.c
    third_party/libopenjpeg/function_list.c
    third_party/libopenjpeg/ht_dec.c
    third_party/libopenjpeg/image.c
    third_party/libopenjpeg/invert.c
    third_party/libopenjpeg/j2k.c
    third_party/libopenjpeg/jp2.c
    third_party/libopenjpeg/mct.c
    third_party/libopenjpeg/mqc.c
    third_party/libopenjpeg/openjpeg.c
    third_party/libopenjpeg/opj_malloc.cc
    third_party/libopenjpeg/pi.c
    third_party/libopenjpeg/sparse_array.c
    third_party/libopenjpeg/t1.c
    third_party/libopenjpeg/t2.c
    third_party/libopenjpeg/tcd.c
    third_party/libopenjpeg/tgt.c
    third_party/libopenjpeg/thread.c
)
target_include_directories(openjpeg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(openjpeg PRIVATE OPJ_STATIC)

# ---------------------------------------------------------------------------
# Core PDFium sources (cross-platform)
# ---------------------------------------------------------------------------
set(PDFIUM_SOURCES
    # fdrm
    core/fdrm/fx_crypt.cpp
    core/fdrm/fx_crypt_aes.cpp
    core/fdrm/fx_crypt_sha.cpp

    # fpdfapi/cmaps
    core/fpdfapi/cmaps/fpdf_cmaps.cpp
    core/fpdfapi/cmaps/CNS1/B5pc-H_0.cpp
    core/fpdfapi/cmaps/CNS1/B5pc-V_0.cpp
    core/fpdfapi/cmaps/CNS1/CNS-EUC-H_0.cpp
    core/fpdfapi/cmaps/CNS1/CNS-EUC-V_0.cpp
    core/fpdfapi/cmaps/CNS1/ETen-B5-H_0.cpp
    core/fpdfapi/cmaps/CNS1/ETen-B5-V_0.cpp
    core/fpdfapi/cmaps/CNS1/ETenms-B5-H_0.cpp
    core/fpdfapi/cmaps/CNS1/ETenms-B5-V_0.cpp
    core/fpdfapi/cmaps/CNS1/HKscs-B5-H_5.cpp
    core/fpdfapi/cmaps/CNS1/HKscs-B5-V_5.cpp
    core/fpdfapi/cmaps/CNS1/UniCNS-UCS2-H_3.cpp
    core/fpdfapi/cmaps/CNS1/UniCNS-UCS2-V_3.cpp
    core/fpdfapi/cmaps/CNS1/UniCNS-UTF16-H_0.cpp
    core/fpdfapi/cmaps/GB1/GB-EUC-H_0.cpp
    core/fpdfapi/cmaps/GB1/GB-EUC-V_0.cpp
    core/fpdfapi/cmaps/GB1/GBK-EUC-H_2.cpp
    core/fpdfapi/cmaps/GB1/GBK-EUC-V_2.cpp
    core/fpdfapi/cmaps/GB1/GBK2K-H_5.cpp
    core/fpdfapi/cmaps/GB1/GBK2K-V_5.cpp
    core/fpdfapi/cmaps/GB1/GBKp-EUC-H_2.cpp
    core/fpdfapi/cmaps/GB1/GBKp-EUC-V_2.cpp
    core/fpdfapi/cmaps/GB1/GBpc-EUC-H_0.cpp
    core/fpdfapi/cmaps/GB1/GBpc-EUC-V_0.cpp
    core/fpdfapi/cmaps/GB1/UniGB-UCS2-H_4.cpp
    core/fpdfapi/cmaps/GB1/UniGB-UCS2-V_4.cpp
    core/fpdfapi/cmaps/Japan1/83pv-RKSJ-H_1.cpp
    core/fpdfapi/cmaps/Japan1/90ms-RKSJ-H_2.cpp
    core/fpdfapi/cmaps/Japan1/90ms-RKSJ-V_2.cpp
    core/fpdfapi/cmaps/Japan1/90msp-RKSJ-H_2.cpp
    core/fpdfapi/cmaps/Japan1/90msp-RKSJ-V_2.cpp
    core/fpdfapi/cmaps/Japan1/90pv-RKSJ-H_1.cpp
    core/fpdfapi/cmaps/Japan1/Add-RKSJ-H_1.cpp
    core/fpdfapi/cmaps/Japan1/Add-RKSJ-V_1.cpp
    core/fpdfapi/cmaps/Japan1/EUC-H_1.cpp
    core/fpdfapi/cmaps/Japan1/EUC-V_1.cpp
    core/fpdfapi/cmaps/Japan1/Ext-RKSJ-H_2.cpp
    core/fpdfapi/cmaps/Japan1/Ext-RKSJ-V_2.cpp
    core/fpdfapi/cmaps/Japan1/H_1.cpp
    core/fpdfapi/cmaps/Japan1/UniJIS-UCS2-H_4.cpp
    core/fpdfapi/cmaps/Japan1/UniJIS-UCS2-HW-H_4.cpp
    core/fpdfapi/cmaps/Japan1/UniJIS-UCS2-HW-V_4.cpp
    core/fpdfapi/cmaps/Japan1/UniJIS-UCS2-V_4.cpp
    core/fpdfapi/cmaps/Japan1/V_1.cpp
    core/fpdfapi/cmaps/Korea1/KSC-EUC-H_0.cpp
    core/fpdfapi/cmaps/Korea1/KSC-EUC-V_0.cpp
    core/fpdfapi/cmaps/Korea1/KSCms-UHC-H_1.cpp
    core/fpdfapi/cmaps/Korea1/KSCms-UHC-HW-H_1.cpp
    core/fpdfapi/cmaps/Korea1/KSCms-UHC-HW-V_1.cpp
    core/fpdfapi/cmaps/Korea1/KSCms-UHC-V_1.cpp
    core/fpdfapi/cmaps/Korea1/KSCpc-EUC-H_0.cpp
    core/fpdfapi/cmaps/Korea1/UniKS-UCS2-H_1.cpp
    core/fpdfapi/cmaps/Korea1/UniKS-UCS2-V_1.cpp
    core/fpdfapi/cmaps/Korea1/UniKS-UTF16-H_0.cpp

    # fpdfapi/edit
    core/fpdfapi/edit/cpdf_creator.cpp
    core/fpdfapi/edit/cpdf_npagetooneexporter.cpp
    core/fpdfapi/edit/cpdf_pagecontentgenerator.cpp
    core/fpdfapi/edit/cpdf_pagecontentmanager.cpp
    core/fpdfapi/edit/cpdf_pageexporter.cpp
    core/fpdfapi/edit/cpdf_pageorganizer.cpp
    core/fpdfapi/edit/cpdf_stringarchivestream.cpp
    core/fpdfapi/edit/cpdf_contentstream_write_utils.cpp

    # fpdfapi/font
    core/fpdfapi/font/cfx_cttgsubtable.cpp
    core/fpdfapi/font/cfx_stockfontarray.cpp
    core/fpdfapi/font/cpdf_cid2unicodemap.cpp
    core/fpdfapi/font/cpdf_cidfont.cpp
    core/fpdfapi/font/cpdf_cmap.cpp
    core/fpdfapi/font/cpdf_cmapparser.cpp
    core/fpdfapi/font/cpdf_font.cpp
    core/fpdfapi/font/cpdf_fontencoding.cpp
    core/fpdfapi/font/cpdf_fontglobals.cpp
    core/fpdfapi/font/cpdf_simplefont.cpp
    core/fpdfapi/font/cpdf_tounicodemap.cpp
    core/fpdfapi/font/cpdf_truetypefont.cpp
    core/fpdfapi/font/cpdf_type1font.cpp
    core/fpdfapi/font/cpdf_type3char.cpp
    core/fpdfapi/font/cpdf_type3font.cpp

    # fpdfapi/page
    core/fpdfapi/page/cpdf_allstates.cpp
    core/fpdfapi/page/cpdf_annotcontext.cpp
    core/fpdfapi/page/cpdf_basedcs.cpp
    core/fpdfapi/page/cpdf_clippath.cpp
    core/fpdfapi/page/cpdf_color.cpp
    core/fpdfapi/page/cpdf_colorspace.cpp
    core/fpdfapi/page/cpdf_colorstate.cpp
    core/fpdfapi/page/cpdf_contentmarkitem.cpp
    core/fpdfapi/page/cpdf_contentmarks.cpp
    core/fpdfapi/page/cpdf_contentparser.cpp
    core/fpdfapi/page/cpdf_devicecs.cpp
    core/fpdfapi/page/cpdf_dib.cpp
    core/fpdfapi/page/cpdf_docpagedata.cpp
    core/fpdfapi/page/cpdf_expintfunc.cpp
    core/fpdfapi/page/cpdf_form.cpp
    core/fpdfapi/page/cpdf_formobject.cpp
    core/fpdfapi/page/cpdf_function.cpp
    core/fpdfapi/page/cpdf_generalstate.cpp
    core/fpdfapi/page/cpdf_graphicstates.cpp
    core/fpdfapi/page/cpdf_iccprofile.cpp
    core/fpdfapi/page/cpdf_image.cpp
    core/fpdfapi/page/cpdf_imageloader.cpp
    core/fpdfapi/page/cpdf_imageobject.cpp
    core/fpdfapi/page/cpdf_indexedcs.cpp
    core/fpdfapi/page/cpdf_meshstream.cpp
    core/fpdfapi/page/cpdf_occontext.cpp
    core/fpdfapi/page/cpdf_page.cpp
    core/fpdfapi/page/cpdf_pageimagecache.cpp
    core/fpdfapi/page/cpdf_pagemodule.cpp
    core/fpdfapi/page/cpdf_pageobject.cpp
    core/fpdfapi/page/cpdf_pageobjectholder.cpp
    core/fpdfapi/page/cpdf_path.cpp
    core/fpdfapi/page/cpdf_pathobject.cpp
    core/fpdfapi/page/cpdf_pattern.cpp
    core/fpdfapi/page/cpdf_patterncs.cpp
    core/fpdfapi/page/cpdf_psengine.cpp
    core/fpdfapi/page/cpdf_psfunc.cpp
    core/fpdfapi/page/cpdf_sampledfunc.cpp
    core/fpdfapi/page/cpdf_shadingobject.cpp
    core/fpdfapi/page/cpdf_shadingpattern.cpp
    core/fpdfapi/page/cpdf_stitchfunc.cpp
    core/fpdfapi/page/cpdf_streamcontentparser.cpp
    core/fpdfapi/page/cpdf_streamparser.cpp
    core/fpdfapi/page/cpdf_textobject.cpp
    core/fpdfapi/page/cpdf_textstate.cpp
    core/fpdfapi/page/cpdf_tilingpattern.cpp
    core/fpdfapi/page/cpdf_transferfunc.cpp
    core/fpdfapi/page/cpdf_transferfuncdib.cpp
    core/fpdfapi/page/cpdf_transparency.cpp
    core/fpdfapi/page/jpx_decode_conversion.cpp

    # fpdfapi/parser
    core/fpdfapi/parser/cfdf_document.cpp
    core/fpdfapi/parser/cpdf_array.cpp
    core/fpdfapi/parser/cpdf_boolean.cpp
    core/fpdfapi/parser/cpdf_cross_ref_avail.cpp
    core/fpdfapi/parser/cpdf_cross_ref_table.cpp
    core/fpdfapi/parser/cpdf_crypto_handler.cpp
    core/fpdfapi/parser/cpdf_data_avail.cpp
    core/fpdfapi/parser/cpdf_dictionary.cpp
    core/fpdfapi/parser/cpdf_document.cpp
    core/fpdfapi/parser/cpdf_encryptor.cpp
    core/fpdfapi/parser/cpdf_flateencoder.cpp
    core/fpdfapi/parser/cpdf_hint_tables.cpp
    core/fpdfapi/parser/cpdf_indirect_object_holder.cpp
    core/fpdfapi/parser/cpdf_linearized_header.cpp
    core/fpdfapi/parser/cpdf_name.cpp
    core/fpdfapi/parser/cpdf_null.cpp
    core/fpdfapi/parser/cpdf_number.cpp
    core/fpdfapi/parser/cpdf_object.cpp
    core/fpdfapi/parser/cpdf_object_avail.cpp
    core/fpdfapi/parser/cpdf_object_stream.cpp
    core/fpdfapi/parser/cpdf_object_walker.cpp
    core/fpdfapi/parser/cpdf_page_object_avail.cpp
    core/fpdfapi/parser/cpdf_parser.cpp
    core/fpdfapi/parser/cpdf_read_validator.cpp
    core/fpdfapi/parser/cpdf_reference.cpp
    core/fpdfapi/parser/cpdf_security_handler.cpp
    core/fpdfapi/parser/cpdf_simple_parser.cpp
    core/fpdfapi/parser/cpdf_stream.cpp
    core/fpdfapi/parser/cpdf_stream_acc.cpp
    core/fpdfapi/parser/cpdf_string.cpp
    core/fpdfapi/parser/cpdf_syntax_parser.cpp
    core/fpdfapi/parser/fpdf_parser_decode.cpp
    core/fpdfapi/parser/fpdf_parser_utility.cpp
    core/fpdfapi/parser/object_tree_traversal_util.cpp

    # fpdfapi/render
    core/fpdfapi/render/cpdf_devicebuffer.cpp
    core/fpdfapi/render/cpdf_docrenderdata.cpp
    core/fpdfapi/render/cpdf_imagerenderer.cpp
    core/fpdfapi/render/cpdf_pagerendercontext.cpp
    core/fpdfapi/render/cpdf_progressiverenderer.cpp
    core/fpdfapi/render/cpdf_rendercontext.cpp
    core/fpdfapi/render/cpdf_renderoptions.cpp
    core/fpdfapi/render/cpdf_rendershading.cpp
    core/fpdfapi/render/cpdf_renderstatus.cpp
    core/fpdfapi/render/cpdf_rendertiling.cpp
    core/fpdfapi/render/cpdf_textrenderer.cpp
    core/fpdfapi/render/cpdf_type3cache.cpp
    core/fpdfapi/render/cpdf_type3glyphmap.cpp

    # fpdfdoc
    core/fpdfdoc/cpdf_aaction.cpp
    core/fpdfdoc/cpdf_action.cpp
    core/fpdfdoc/cpdf_annot.cpp
    core/fpdfdoc/cpdf_annotlist.cpp
    core/fpdfdoc/cpdf_apsettings.cpp
    core/fpdfdoc/cpdf_bafontmap.cpp
    core/fpdfdoc/cpdf_bookmark.cpp
    core/fpdfdoc/cpdf_bookmarktree.cpp
    core/fpdfdoc/cpdf_color_utils.cpp
    core/fpdfdoc/cpdf_defaultappearance.cpp
    core/fpdfdoc/cpdf_dest.cpp
    core/fpdfdoc/cpdf_filespec.cpp
    core/fpdfdoc/cpdf_formcontrol.cpp
    core/fpdfdoc/cpdf_formfield.cpp
    core/fpdfdoc/cpdf_generateap.cpp
    core/fpdfdoc/cpdf_icon.cpp
    core/fpdfdoc/cpdf_iconfit.cpp
    core/fpdfdoc/cpdf_interactiveform.cpp
    core/fpdfdoc/cpdf_link.cpp
    core/fpdfdoc/cpdf_linklist.cpp
    core/fpdfdoc/cpdf_metadata.cpp
    core/fpdfdoc/cpdf_nametree.cpp
    core/fpdfdoc/cpdf_numbertree.cpp
    core/fpdfdoc/cpdf_pagelabel.cpp
    core/fpdfdoc/cpdf_structelement.cpp
    core/fpdfdoc/cpdf_structtree.cpp
    core/fpdfdoc/cpdf_viewerpreferences.cpp
    core/fpdfdoc/cpvt_fontmap.cpp
    core/fpdfdoc/cpvt_section.cpp
    core/fpdfdoc/cpvt_variabletext.cpp
    core/fpdfdoc/cpvt_wordinfo.cpp

    # fpdftext
    core/fpdftext/cpdf_linkextract.cpp
    core/fpdftext/cpdf_textpage.cpp
    core/fpdftext/cpdf_textpagefind.cpp
    core/fpdftext/unicodenormalizationdata.cpp

    # fxcodec
    core/fxcodec/basic/basicmodule.cpp
    core/fxcodec/data_and_bytes_consumed.cpp
    core/fxcodec/fax/faxmodule.cpp
    core/fxcodec/flate/flatemodule.cpp
    core/fxcodec/fx_codec.cpp
    core/fxcodec/icc/icc_transform.cpp
    core/fxcodec/jbig2/JBig2_ArithDecoder.cpp
    core/fxcodec/jbig2/JBig2_ArithIntDecoder.cpp
    core/fxcodec/jbig2/JBig2_BitStream.cpp
    core/fxcodec/jbig2/JBig2_Context.cpp
    core/fxcodec/jbig2/JBig2_DocumentContext.cpp
    core/fxcodec/jbig2/JBig2_GrdProc.cpp
    core/fxcodec/jbig2/JBig2_GrrdProc.cpp
    core/fxcodec/jbig2/JBig2_HtrdProc.cpp
    core/fxcodec/jbig2/JBig2_HuffmanDecoder.cpp
    core/fxcodec/jbig2/JBig2_HuffmanTable.cpp
    core/fxcodec/jbig2/JBig2_Image.cpp
    core/fxcodec/jbig2/JBig2_PatternDict.cpp
    core/fxcodec/jbig2/JBig2_PddProc.cpp
    core/fxcodec/jbig2/JBig2_SddProc.cpp
    core/fxcodec/jbig2/JBig2_Segment.cpp
    core/fxcodec/jbig2/JBig2_SymbolDict.cpp
    core/fxcodec/jbig2/JBig2_TrdProc.cpp
    core/fxcodec/jbig2/jbig2_decoder.cpp
    core/fxcodec/jpeg/jpegmodule.cpp
    core/fxcodec/jpeg/jpeg_common.c
    core/fxcodec/jpx/cjpx_decoder.cpp
    core/fxcodec/jpx/jpx_decode_utils.cpp
    core/fxcodec/scanlinedecoder.cpp

    # fxcrt (cross-platform)
    core/fxcrt/binary_buffer.cpp
    core/fxcrt/bytestring.cpp
    core/fxcrt/cfx_bitstream.cpp
    core/fxcrt/cfx_datetime.cpp
    core/fxcrt/cfx_memorystream.cpp
    core/fxcrt/cfx_fileaccess_stream.cpp
    core/fxcrt/cfx_read_only_container_stream.cpp
    core/fxcrt/cfx_read_only_span_stream.cpp
    core/fxcrt/cfx_seekablestreamproxy.cpp
    core/fxcrt/mapped_data_bytes.cpp
    core/fxcrt/cfx_timer.cpp
    core/fxcrt/debug/alias.cc
    core/fxcrt/fx_bidi.cpp
    core/fxcrt/fx_codepage.cpp
    core/fxcrt/fx_coordinates.cpp
    core/fxcrt/fx_extension.cpp
    core/fxcrt/fx_memory.cpp
    core/fxcrt/fx_number.cpp
    core/fxcrt/fx_random.cpp
    core/fxcrt/fx_stream.cpp
    core/fxcrt/fx_string.cpp
    core/fxcrt/fx_system.cpp
    core/fxcrt/fx_unicode.cpp
    core/fxcrt/observed_ptr.cpp
    core/fxcrt/string_data_template.cpp
    core/fxcrt/string_template.cpp
    core/fxcrt/widestring.cpp
    core/fxcrt/widetext_buffer.cpp
    core/fxcrt/xml/cfx_xmlchardata.cpp
    core/fxcrt/xml/cfx_xmldocument.cpp
    core/fxcrt/xml/cfx_xmlelement.cpp
    core/fxcrt/xml/cfx_xmlinstruction.cpp
    core/fxcrt/xml/cfx_xmlnode.cpp
    core/fxcrt/xml/cfx_xmlparser.cpp
    core/fxcrt/xml/cfx_xmltext.cpp
    core/fxcrt/fx_memory_malloc.cpp

    # fxge (cross-platform)
    core/fxge/agg/cfx_agg_bitmapcomposer.cpp
    core/fxge/agg/cfx_agg_cliprgn.cpp
    core/fxge/agg/cfx_agg_devicedriver.cpp
    core/fxge/agg/cfx_agg_imagerenderer.cpp
    core/fxge/calculate_pitch.cpp
    core/fxge/cfx_color.cpp
    core/fxge/cfx_defaultrenderdevice.cpp
    core/fxge/cfx_drawutils.cpp
    core/fxge/cfx_face.cpp
    core/fxge/cfx_folderfontinfo.cpp
    core/fxge/cfx_font.cpp
    core/fxge/cfx_fontmapper.cpp
    core/fxge/cfx_fontmgr.cpp
    core/fxge/cfx_gemodule.cpp
    core/fxge/cfx_glyphbitmap.cpp
    core/fxge/cfx_glyphcache.cpp
    core/fxge/cfx_graphstate.cpp
    core/fxge/cfx_graphstatedata.cpp
    core/fxge/cfx_path.cpp
    core/fxge/cfx_renderdevice.cpp
    core/fxge/cfx_substfont.cpp
    core/fxge/cfx_unicodeencoding.cpp
    core/fxge/dib/blend.cpp
    core/fxge/dib/cfx_bitmapstorer.cpp
    core/fxge/dib/cfx_cmyk_to_srgb.cpp
    core/fxge/dib/cfx_dibbase.cpp
    core/fxge/dib/cfx_dibitmap.cpp
    core/fxge/dib/cfx_imagestretcher.cpp
    core/fxge/dib/cfx_imagetransformer.cpp
    core/fxge/dib/cfx_scanlinecompositor.cpp
    core/fxge/dib/cstretchengine.cpp
    core/fxge/dib/fx_dib.cpp
    core/fxge/fontdata/chromefontdata/FoxitDingbats.cpp
    core/fxge/fontdata/chromefontdata/FoxitFixed.cpp
    core/fxge/fontdata/chromefontdata/FoxitFixedBold.cpp
    core/fxge/fontdata/chromefontdata/FoxitFixedBoldItalic.cpp
    core/fxge/fontdata/chromefontdata/FoxitFixedItalic.cpp
    core/fxge/fontdata/chromefontdata/FoxitSans.cpp
    core/fxge/fontdata/chromefontdata/FoxitSansBold.cpp
    core/fxge/fontdata/chromefontdata/FoxitSansBoldItalic.cpp
    core/fxge/fontdata/chromefontdata/FoxitSansItalic.cpp
    core/fxge/fontdata/chromefontdata/FoxitSansMM.cpp
    core/fxge/fontdata/chromefontdata/FoxitSerif.cpp
    core/fxge/fontdata/chromefontdata/FoxitSerifBold.cpp
    core/fxge/fontdata/chromefontdata/FoxitSerifBoldItalic.cpp
    core/fxge/fontdata/chromefontdata/FoxitSerifItalic.cpp
    core/fxge/fontdata/chromefontdata/FoxitSerifMM.cpp
    core/fxge/fontdata/chromefontdata/FoxitSymbol.cpp
    core/fxge/freetype/fx_freetype.cpp
    core/fxge/fx_font.cpp
    core/fxge/renderdevicedriver_iface.cpp
    core/fxge/text_char_pos.cpp
    core/fxge/text_glyph_pos.cpp

    # fpdfsdk
    fpdfsdk/cpdfsdk_annot.cpp
    fpdfsdk/cpdfsdk_annotiteration.cpp
    fpdfsdk/cpdfsdk_annotiterator.cpp
    fpdfsdk/cpdfsdk_appstream.cpp
    fpdfsdk/cpdfsdk_baannot.cpp
    fpdfsdk/cpdfsdk_customaccess.cpp
    fpdfsdk/cpdfsdk_filewriteadapter.cpp
    fpdfsdk/cpdfsdk_formfillenvironment.cpp
    fpdfsdk/cpdfsdk_helpers.cpp
    fpdfsdk/cpdfsdk_interactiveform.cpp
    fpdfsdk/cpdfsdk_pageview.cpp
    fpdfsdk/cpdfsdk_pauseadapter.cpp
    fpdfsdk/cpdfsdk_renderpage.cpp
    fpdfsdk/cpdfsdk_widget.cpp
    fpdfsdk/fpdf_annot.cpp
    fpdfsdk/fpdf_attachment.cpp
    fpdfsdk/fpdf_catalog.cpp
    fpdfsdk/fpdf_dataavail.cpp
    fpdfsdk/fpdf_doc.cpp
    fpdfsdk/fpdf_editimg.cpp
    fpdfsdk/fpdf_editpage.cpp
    fpdfsdk/fpdf_editpath.cpp
    fpdfsdk/fpdf_edittext.cpp
    fpdfsdk/fpdf_ext.cpp
    fpdfsdk/fpdf_flatten.cpp
    fpdfsdk/fpdf_formfill.cpp
    fpdfsdk/fpdf_javascript.cpp
    fpdfsdk/fpdf_ppo.cpp
    fpdfsdk/fpdf_progressive.cpp
    fpdfsdk/fpdf_save.cpp
    fpdfsdk/fpdf_searchex.cpp
    fpdfsdk/fpdf_signature.cpp
    fpdfsdk/fpdf_structtree.cpp
    fpdfsdk/fpdf_sysfontinfo.cpp
    fpdfsdk/fpdf_text.cpp
    fpdfsdk/fpdf_thumbnail.cpp
    fpdfsdk/fpdf_transformpage.cpp
    fpdfsdk/fpdf_view.cpp

    # fpdfsdk/formfiller
    fpdfsdk/formfiller/cffl_button.cpp
    fpdfsdk/formfiller/cffl_checkbox.cpp
    fpdfsdk/formfiller/cffl_combobox.cpp
    fpdfsdk/formfiller/cffl_fieldaction.cpp
    fpdfsdk/formfiller/cffl_formfield.cpp
    fpdfsdk/formfiller/cffl_interactiveformfiller.cpp
    fpdfsdk/formfiller/cffl_listbox.cpp
    fpdfsdk/formfiller/cffl_perwindowdata.cpp
    fpdfsdk/formfiller/cffl_pushbutton.cpp
    fpdfsdk/formfiller/cffl_radiobutton.cpp
    fpdfsdk/formfiller/cffl_textfield.cpp
    fpdfsdk/formfiller/cffl_textobject.cpp

    # fpdfsdk/pwl
    fpdfsdk/pwl/cpwl_button.cpp
    fpdfsdk/pwl/cpwl_caret.cpp
    fpdfsdk/pwl/cpwl_cbbutton.cpp
    fpdfsdk/pwl/cpwl_cblistbox.cpp
    fpdfsdk/pwl/cpwl_combo_box.cpp
    fpdfsdk/pwl/cpwl_edit.cpp
    fpdfsdk/pwl/cpwl_edit_impl.cpp
    fpdfsdk/pwl/cpwl_list_box.cpp
    fpdfsdk/pwl/cpwl_list_ctrl.cpp
    fpdfsdk/pwl/cpwl_sbbutton.cpp
    fpdfsdk/pwl/cpwl_scroll_bar.cpp
    fpdfsdk/pwl/cpwl_special_button.cpp
    fpdfsdk/pwl/cpwl_wnd.cpp

    # fxjs (stubs — no V8)
    fxjs/cjs_event_context_stub.cpp
    fxjs/cjs_runtimestub.cpp
    fxjs/ijs_runtime.cpp
)

# Platform-specific sources
# Use headless mode for static builds (no system font dependencies)
option(PDFIUM_HEADLESS "Build without platform font rendering (static build)" ON)

if(WIN32)
    list(APPEND PDFIUM_SOURCES
        core/fxcrt/code_point_view.cpp
        core/fxcrt/cfx_fileaccess_windows.cpp
        core/fxcrt/fx_folder_windows.cpp
        core/fxcrt/win/win_util.cc
        core/fxge/cfx_windowsrenderdevice.cpp
        core/fxge/win32/cfx_psfonttracker.cpp
        core/fxge/win32/cfx_psrenderer.cpp
        core/fxge/win32/cgdi_device_driver.cpp
        core/fxge/win32/cgdi_display_driver.cpp
        core/fxge/win32/cgdi_plus_ext.cpp
        core/fxge/win32/cgdi_printer_driver.cpp
        core/fxge/win32/cps_printer_driver.cpp
        core/fxge/win32/cpsoutput.cpp
        core/fxge/win32/ctext_only_printer_driver.cpp
        core/fxge/win32/cwin32_platform.cpp
        core/fpdfapi/render/cpdf_scaledrenderbuffer.cpp
        core/fpdfapi/render/cpdf_windowsrenderdevice.cpp
    )
else()
    # POSIX (Linux/macOS) - use headless implementation for static builds
    list(APPEND PDFIUM_SOURCES
        core/fxcrt/cfx_fileaccess_posix.cpp
        core/fxcrt/fx_folder_posix.cpp
    )
    if(PDFIUM_HEADLESS)
        # Headless: no system font rendering, no framework dependencies
        list(APPEND PDFIUM_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/../../ports/pdfium/fx_headless_impl.cpp
        )
        add_definitions(-DPDFIUM_HEADLESS)
    elseif(APPLE)
        list(APPEND PDFIUM_SOURCES
            core/fxge/apple/fx_apple_impl.cpp
            core/fxge/apple/fx_apple_platform.cpp
            core/fxge/apple/fx_quartz_device.cpp
        )
    else()
        list(APPEND PDFIUM_SOURCES
            core/fxge/linux/fx_linux_impl.cpp
        )
    endif()
endif()

# ---------------------------------------------------------------------------
# PDFium library target
# ---------------------------------------------------------------------------
add_library(pdfium STATIC ${PDFIUM_SOURCES})

target_include_directories(pdfium PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FREETYPE_INCLUDE_DIR}
    ${ICU_INCLUDE_DIRS}
    ${JPEG_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIRS}
)

# fast_float is header-only, bundled in PDFium's third_party
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/fast_float/fast_float/include")
    target_include_directories(pdfium PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fast_float/fast_float/include
    )
endif()

target_compile_definitions(pdfium PRIVATE
    COMPONENT_BUILD
    FPDF_IMPLEMENTATION
    OPJ_STATIC
)

target_link_libraries(pdfium PRIVATE
    agg lcms openjpeg
    freetype_imported
)

# macOS: link CoreGraphics and CoreFoundation only for non-headless Apple builds
if(APPLE AND NOT PDFIUM_HEADLESS)
    find_library(COREGRAPHICS_FRAMEWORK CoreGraphics)
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
    target_link_libraries(pdfium PRIVATE
        ${COREGRAPHICS_FRAMEWORK}
        ${COREFOUNDATION_FRAMEWORK}
    )
endif()

install(TARGETS pdfium agg lcms openjpeg
    ARCHIVE DESTINATION lib
)
