//! Filter and sort enums for HuggingFace model search.

use talu::SearchSort;

// ---------------------------------------------------------------------------
// Sort enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SortMode {
    Trending,
    Downloads,
    Likes,
    Recent,
}

impl SortMode {
    pub(super) const ALL: &[SortMode] = &[
        SortMode::Trending,
        SortMode::Downloads,
        SortMode::Likes,
        SortMode::Recent,
    ];

    pub(super) fn label(self) -> &'static str {
        match self {
            SortMode::Trending => "Trending",
            SortMode::Downloads => "Downloads",
            SortMode::Likes => "Likes",
            SortMode::Recent => "Recent",
        }
    }

    pub(super) fn to_api(self) -> SearchSort {
        match self {
            SortMode::Trending => SearchSort::Trending,
            SortMode::Downloads => SearchSort::Downloads,
            SortMode::Likes => SearchSort::Likes,
            SortMode::Recent => SearchSort::LastModified,
        }
    }
}

// ---------------------------------------------------------------------------
// Size filter (client-side, applied after search)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SizeFilter {
    Max1B,
    Max2B,
    Max4B,
    Max8B,
    Max16B,
    Max32B,
    Max64B,
    Max128B,
    Max512B,
    All,
}

impl SizeFilter {
    /// Ordered 1-9 then 0.  Index 0-8 map to keys 1-9, index 9 maps to key 0.
    pub(super) const ALL: &[SizeFilter] = &[
        SizeFilter::Max1B,
        SizeFilter::Max2B,
        SizeFilter::Max4B,
        SizeFilter::Max8B,
        SizeFilter::Max16B,
        SizeFilter::Max32B,
        SizeFilter::Max64B,
        SizeFilter::Max128B,
        SizeFilter::Max512B,
        SizeFilter::All,
    ];

    pub(super) fn label(self) -> &'static str {
        match self {
            SizeFilter::Max1B => "\u{2264}1B",
            SizeFilter::Max2B => "\u{2264}2B",
            SizeFilter::Max4B => "\u{2264}4B",
            SizeFilter::Max8B => "\u{2264}8B",
            SizeFilter::Max16B => "\u{2264}16B",
            SizeFilter::Max32B => "\u{2264}32B",
            SizeFilter::Max64B => "\u{2264}64B",
            SizeFilter::Max128B => "\u{2264}128B",
            SizeFilter::Max512B => "\u{2264}512B",
            SizeFilter::All => "Any",
        }
    }

    pub(super) fn max_params(self) -> Option<i64> {
        match self {
            SizeFilter::Max1B => Some(1_500_000_000),
            SizeFilter::Max2B => Some(2_500_000_000),
            SizeFilter::Max4B => Some(4_500_000_000),
            SizeFilter::Max8B => Some(8_500_000_000),
            SizeFilter::Max16B => Some(16_500_000_000),
            SizeFilter::Max32B => Some(32_500_000_000),
            SizeFilter::Max64B => Some(65_000_000_000),
            SizeFilter::Max128B => Some(130_000_000_000),
            SizeFilter::Max512B => Some(520_000_000_000),
            SizeFilter::All => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Task filter (maps to HF pipeline_tag)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TaskFilter {
    TextGeneration,
    Multimodal,
    ImageToText,
    TextToImage,
    TextToSpeech,
    SentenceSimilarity,
    All,
}

impl TaskFilter {
    pub(super) const ALL: &[TaskFilter] = &[
        TaskFilter::TextGeneration,
        TaskFilter::Multimodal,
        TaskFilter::ImageToText,
        TaskFilter::TextToImage,
        TaskFilter::TextToSpeech,
        TaskFilter::SentenceSimilarity,
        TaskFilter::All,
    ];

    pub(super) fn label(self) -> &'static str {
        match self {
            TaskFilter::TextGeneration => "Text Generation",
            TaskFilter::Multimodal => "Multimodal",
            TaskFilter::ImageToText => "Image to Text",
            TaskFilter::TextToImage => "Text to Image",
            TaskFilter::TextToSpeech => "Text to Speech",
            TaskFilter::SentenceSimilarity => "Sentence Similarity",
            TaskFilter::All => "Any",
        }
    }

    pub(super) fn to_api_string(self) -> Option<&'static str> {
        match self {
            TaskFilter::TextGeneration => Some("text-generation"),
            TaskFilter::Multimodal => Some("image-text-to-text"),
            TaskFilter::ImageToText => Some("image-to-text"),
            TaskFilter::TextToImage => Some("text-to-image"),
            TaskFilter::TextToSpeech => Some("text-to-speech"),
            TaskFilter::SentenceSimilarity => Some("sentence-similarity"),
            TaskFilter::All => None,
        }
    }

    /// Additional pipeline tags that imply the same capability.
    /// Multimodal models tagged `image-text-to-text` generate text from
    /// images+text, so they match both text-generation and image-to-text.
    pub(super) fn extra_api_tags(self) -> &'static [&'static str] {
        match self {
            TaskFilter::TextGeneration => &["image-text-to-text"],
            TaskFilter::Multimodal => &["any-to-any"],
            TaskFilter::ImageToText => &["image-text-to-text"],
            _ => &[],
        }
    }
}

// ---------------------------------------------------------------------------
// Library filter (AND-combined with task in the HF filter= param)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LibraryFilter {
    Safetensors,
    Transformers,
    Mlx,
    SentenceTransformers,
    All,
}

impl LibraryFilter {
    pub(super) const ALL: &[LibraryFilter] = &[
        LibraryFilter::Safetensors,
        LibraryFilter::Transformers,
        LibraryFilter::Mlx,
        LibraryFilter::SentenceTransformers,
        LibraryFilter::All,
    ];

    pub(super) fn label(self) -> &'static str {
        match self {
            LibraryFilter::Safetensors => "safetensors",
            LibraryFilter::Transformers => "transformers",
            LibraryFilter::Mlx => "mlx",
            LibraryFilter::SentenceTransformers => "sentence-transformers",
            LibraryFilter::All => "Any",
        }
    }

    pub(super) fn to_api_string(self) -> Option<&'static str> {
        match self {
            LibraryFilter::Safetensors => Some("safetensors"),
            LibraryFilter::Transformers => Some("transformers"),
            LibraryFilter::Mlx => Some("mlx"),
            LibraryFilter::SentenceTransformers => Some("sentence-transformers"),
            LibraryFilter::All => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Date filter (client-side, applied after search)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DateFilter {
    Last30d,
    Last60d,
    Last90d,
    Last120d,
    Last180d,
    Last360d,
    Last2y,
    All,
}

impl DateFilter {
    /// 1-7 map to time ranges, 0 = Any.
    pub(super) const ALL: &[DateFilter] = &[
        DateFilter::Last30d,
        DateFilter::Last60d,
        DateFilter::Last90d,
        DateFilter::Last120d,
        DateFilter::Last180d,
        DateFilter::Last360d,
        DateFilter::Last2y,
        DateFilter::All,
    ];

    pub(super) fn label(self) -> &'static str {
        match self {
            DateFilter::Last30d => "\u{2264}30d",
            DateFilter::Last60d => "\u{2264}60d",
            DateFilter::Last90d => "\u{2264}90d",
            DateFilter::Last120d => "\u{2264}120d",
            DateFilter::Last180d => "\u{2264}180d",
            DateFilter::Last360d => "\u{2264}1y",
            DateFilter::Last2y => "\u{2264}2y",
            DateFilter::All => "Any",
        }
    }

    /// Returns the cutoff as days-ago, or None for All.
    pub(super) fn max_age_days(self) -> Option<u64> {
        match self {
            DateFilter::Last30d => Some(30),
            DateFilter::Last60d => Some(60),
            DateFilter::Last90d => Some(90),
            DateFilter::Last120d => Some(120),
            DateFilter::Last180d => Some(180),
            DateFilter::Last360d => Some(360),
            DateFilter::Last2y => Some(730),
            DateFilter::All => None,
        }
    }
}
