//! HfSearchSelector struct, PickerMode, and all logic (non-draw) methods.

use std::collections::HashSet;
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::widgets::TableState;

use talu::{HfSearchResult, SearchDirection, SearchSort};

use super::filters::*;
use super::{days_to_ymd, LabelPositions, SearchResponse, DISPLAY_LIMIT, FETCH_LIMIT};
use crate::pin_store::PinStore;

// ---------------------------------------------------------------------------
// Picker overlay (pulldown-style selection)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PickerMode {
    Task,
    Size,
    Date,
    Library,
    Sort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TabMode {
    Search,
    Pinned,
}

impl TabMode {
    pub(super) fn toggle(self) -> Self {
        match self {
            TabMode::Search => TabMode::Pinned,
            TabMode::Pinned => TabMode::Search,
        }
    }
}

impl PickerMode {
    /// Cycle order for Tab key.
    pub(super) const CYCLE: &[PickerMode] = &[
        PickerMode::Task,
        PickerMode::Size,
        PickerMode::Date,
        PickerMode::Library,
        PickerMode::Sort,
    ];

    /// Return the next picker in Tab cycle (wraps around).
    pub(super) fn next(self) -> PickerMode {
        let pos = Self::CYCLE.iter().position(|&m| m == self).unwrap_or(0);
        Self::CYCLE[(pos + 1) % Self::CYCLE.len()]
    }

    /// Return the previous picker in Tab cycle (wraps around).
    pub(super) fn prev(self) -> PickerMode {
        let pos = Self::CYCLE.iter().position(|&m| m == self).unwrap_or(0);
        Self::CYCLE[(pos + Self::CYCLE.len() - 1) % Self::CYCLE.len()]
    }

    /// Number of items in this picker.
    pub(super) fn item_count(self) -> usize {
        match self {
            PickerMode::Task => TaskFilter::ALL.len(),
            PickerMode::Size => SizeFilter::ALL.len(),
            PickerMode::Date => DateFilter::ALL.len(),
            PickerMode::Library => LibraryFilter::ALL.len(),
            PickerMode::Sort => SortMode::ALL.len(),
        }
    }

    /// Index of the currently active item for this picker.
    pub(super) fn active_index(self, sel: &HfSearchSelector) -> usize {
        match self {
            PickerMode::Task => TaskFilter::ALL
                .iter()
                .position(|&v| v == sel.task_filter)
                .unwrap_or(0),
            PickerMode::Size => SizeFilter::ALL
                .iter()
                .position(|&v| v == sel.size_filter)
                .unwrap_or(0),
            PickerMode::Date => DateFilter::ALL
                .iter()
                .position(|&v| v == sel.date_filter)
                .unwrap_or(0),
            PickerMode::Library => LibraryFilter::ALL
                .iter()
                .position(|&v| v == sel.library_filter)
                .unwrap_or(0),
            PickerMode::Sort => SortMode::ALL
                .iter()
                .position(|&v| v == sel.sort_mode)
                .unwrap_or(0),
        }
    }
}

// ---------------------------------------------------------------------------
// HfSearchSelector
// ---------------------------------------------------------------------------

pub(super) struct HfSearchSelector {
    pub(super) query: String,
    pub(super) results: Vec<HfSearchResult>,
    pub(super) filtered: Vec<HfSearchResult>,
    /// Query string that produced `results` (last successful API response).
    pub(super) cached_query: String,
    /// Model IDs that are already downloaded locally.
    pub(super) cached_ids: HashSet<String>,
    /// Profile-local pinned model IDs loaded from `<bucket>/meta.sqlite`.
    pub(super) pinned_ids: HashSet<String>,
    /// Pinned model list in display order (most recently pinned first).
    pub(super) pinned_models: Vec<String>,
    /// Current pinned list view filtered by query text.
    pub(super) pinned_filtered: Vec<String>,
    pub(super) active_tab: TabMode,
    pub(super) pin_store: PinStore,
    pub(super) state: TableState,
    pub(super) sort_mode: SortMode,
    pub(super) task_filter: TaskFilter,
    pub(super) library_filter: LibraryFilter,
    pub(super) size_filter: SizeFilter,
    pub(super) date_filter: DateFilter,
    pub(super) picker: Option<PickerMode>,
    pub(super) picker_cursor: usize,
    pub(super) label_pos: LabelPositions,
    pub(super) loading: bool,
    /// Number of API responses still pending for the current generation.
    pub(super) pending: u8,
    /// Whether the primary result for the current generation has arrived.
    pub(super) primary_arrived: bool,
    /// Stashed secondary results if they arrive before the primary.
    pub(super) stashed_secondary: Vec<Vec<HfSearchResult>>,
    /// Transient error shown as a floating bar; auto-dismissed after a few seconds.
    pub(super) error_msg: Option<(String, Instant)>,
    pub(super) generation: u64,
    pub(super) last_keystroke: Instant,
    pub(super) query_dirty: bool,
    pub(super) last_saved: Option<String>,
    pub(super) rx: mpsc::Receiver<SearchResponse>,
    pub(super) tx: mpsc::Sender<SearchResponse>,
}

impl HfSearchSelector {
    pub(super) fn new(pin_db_path: &Path) -> Result<Self> {
        let (tx, rx) = mpsc::channel();
        // Load cached model IDs once at startup.
        let cached_ids = talu::repo::repo_list_models(false)
            .map(|models| models.into_iter().map(|m| m.id).collect::<HashSet<_>>())
            .unwrap_or_default();
        let pin_store = PinStore::open(pin_db_path)?;
        let pinned_models = pin_store.list_pinned()?;
        let pinned_ids = pinned_models.iter().cloned().collect::<HashSet<_>>();
        Ok(Self {
            query: String::new(),
            results: Vec::new(),
            filtered: Vec::new(),
            cached_query: String::new(),
            cached_ids,
            pinned_ids,
            pinned_models,
            pinned_filtered: Vec::new(),
            active_tab: TabMode::Search,
            pin_store,
            state: TableState::default(),
            sort_mode: SortMode::Trending,
            task_filter: TaskFilter::TextGeneration,
            library_filter: LibraryFilter::Safetensors,
            size_filter: SizeFilter::Max8B,
            date_filter: DateFilter::Last360d,
            picker: None,
            picker_cursor: 0,
            label_pos: LabelPositions::default(),
            loading: true,
            pending: 0,
            primary_arrived: false,
            stashed_secondary: Vec::new(),
            error_msg: None,
            generation: 0,
            last_keystroke: Instant::now(),
            query_dirty: false,
            last_saved: None,
            rx,
            tx,
        })
    }

    pub(super) fn apply_filters(&mut self) {
        let query_lower = self.query.to_lowercase();
        // If the current query extends the cached query, also filter by
        // model_id substring so the user gets instant narrowing as they type.
        let extra_query = if !query_lower.is_empty()
            && query_lower.starts_with(&self.cached_query.to_lowercase())
        {
            Some(query_lower)
        } else {
            None
        };

        let max_params = self.size_filter.max_params();
        let cutoff = self.date_filter.max_age_days().map(|days| {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let cutoff_secs = secs.saturating_sub(days * 86400);
            let days_since_epoch = cutoff_secs / 86400;
            let (y, m, d) = days_to_ymd(days_since_epoch);
            format!("{:04}-{:02}-{:02}", y, m, d)
        });

        self.filtered = self
            .results
            .iter()
            .filter(|r| {
                // Skip models with unknown parameter count (incomplete uploads).
                if r.params_total <= 0 {
                    return false;
                }
                // Client-side query narrowing.
                if let Some(ref q) = extra_query {
                    if !r.model_id.to_lowercase().contains(q.as_str()) {
                        return false;
                    }
                }
                // Size filter.
                if let Some(limit) = max_params {
                    if r.params_total > limit {
                        return false;
                    }
                }
                if let Some(ref cutoff) = cutoff {
                    if r.last_modified.as_str() < cutoff.as_str() {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        // Client-side sort (Trending keeps server-provided order).
        match self.sort_mode {
            SortMode::Downloads => self.filtered.sort_by(|a, b| b.downloads.cmp(&a.downloads)),
            SortMode::Likes => self.filtered.sort_by(|a, b| b.likes.cmp(&a.likes)),
            SortMode::Recent => self
                .filtered
                .sort_by(|a, b| b.last_modified.cmp(&a.last_modified)),
            SortMode::Trending => {} // server-provided order
        }

        self.filtered.truncate(DISPLAY_LIMIT);
        self.apply_pinned_filters();
        self.reset_selection();
    }

    pub(super) fn apply_pinned_filters(&mut self) {
        let query_lower = self.query.to_lowercase();
        self.pinned_filtered = self
            .pinned_models
            .iter()
            .filter(|id| {
                if query_lower.is_empty() {
                    true
                } else {
                    id.to_lowercase().contains(query_lower.as_str())
                }
            })
            .cloned()
            .collect();
    }

    fn reset_selection(&mut self) {
        if self.current_items_count() > 0 {
            self.state.select(Some(0));
        } else {
            self.state.select(None);
        }
    }

    /// Append secondary (downloads) results after the primary, deduplicating.
    fn merge_secondary(&mut self, secondary: Vec<HfSearchResult>) {
        let existing: HashSet<String> = self.results.iter().map(|r| r.model_id.clone()).collect();
        for r in secondary {
            if !existing.contains(&r.model_id) {
                self.results.push(r);
            }
        }
    }

    pub(super) fn trigger_search(&mut self) {
        self.generation += 1;
        self.loading = true;
        self.error_msg = None;
        self.query_dirty = false;
        self.primary_arrived = false;
        self.stashed_secondary.clear();

        let gen = self.generation;
        let query = self.query.clone();
        let sort = self.sort_mode.to_api();
        let filter = self.task_filter.to_api_string().map(|s| s.to_string());
        let library = self.library_filter.to_api_string().map(|s| s.to_string());
        let token = std::env::var("HF_TOKEN").ok();

        // Primary fetch: user's chosen sort order.
        let tx1 = self.tx.clone();
        let q1 = query.clone();
        let f1 = filter.clone();
        let l1 = library.clone();
        let t1 = token.clone();
        std::thread::spawn(move || {
            let results = talu::repo::repo_search_rich(
                &q1,
                FETCH_LIMIT,
                t1.as_deref(),
                None,
                f1.as_deref(),
                sort,
                SearchDirection::Descending,
                l1.as_deref(),
            );
            let _ = tx1.send(SearchResponse {
                generation: gen,
                is_primary: true,
                results: results.map_err(|e| e.to_string()),
            });
        });

        // Secondary fetch: always by downloads for broader coverage.
        // Skip if the user already sorts by downloads (no point fetching twice).
        let extra_tags = self.task_filter.extra_api_tags();
        let mut extra_count: usize = 0;
        if sort != SearchSort::Downloads {
            extra_count += 1;
            let tx2 = self.tx.clone();
            let query2 = query.clone();
            let filter2 = filter.clone();
            let library2 = library.clone();
            let token2 = token.clone();
            std::thread::spawn(move || {
                let results = talu::repo::repo_search_rich(
                    &query2,
                    FETCH_LIMIT,
                    token2.as_deref(),
                    None,
                    filter2.as_deref(),
                    SearchSort::Downloads,
                    SearchDirection::Descending,
                    library2.as_deref(),
                );
                let _ = tx2.send(SearchResponse {
                    generation: gen,
                    is_primary: false,
                    results: results.map_err(|e| e.to_string()),
                });
            });
        }

        // Extra fetches for related pipeline tags (e.g. multimodal models
        // that also generate text but use a different HF tag).
        for tag in extra_tags {
            extra_count += 1;
            let tx_extra = self.tx.clone();
            let query_extra = query.clone();
            let library_extra = library.clone();
            let token_extra = token.clone();
            let tag = tag.to_string();
            std::thread::spawn(move || {
                let results = talu::repo::repo_search_rich(
                    &query_extra,
                    FETCH_LIMIT,
                    token_extra.as_deref(),
                    None,
                    Some(&tag),
                    SearchSort::Downloads,
                    SearchDirection::Descending,
                    library_extra.as_deref(),
                );
                let _ = tx_extra.send(SearchResponse {
                    generation: gen,
                    is_primary: false,
                    results: results.map_err(|e| e.to_string()),
                });
            });
        }

        self.pending = (1 + extra_count) as u8;
    }

    /// Open a picker overlay, positioning cursor on the currently active item.
    pub(super) fn open_picker(&mut self, mode: PickerMode) {
        self.picker_cursor = mode.active_index(self);
        self.picker = Some(mode);
    }

    /// Apply the picker selection at the current cursor position (keeps picker open).
    pub(super) fn apply_picker(&mut self, mode: PickerMode) {
        match mode {
            PickerMode::Size => {
                if let Some(&val) = SizeFilter::ALL.get(self.picker_cursor) {
                    self.size_filter = val;
                    self.apply_filters();
                }
            }
            PickerMode::Date => {
                if let Some(&val) = DateFilter::ALL.get(self.picker_cursor) {
                    self.date_filter = val;
                    self.apply_filters();
                }
            }
            PickerMode::Sort => {
                if let Some(&val) = SortMode::ALL.get(self.picker_cursor) {
                    self.sort_mode = val;
                    if val == SortMode::Trending {
                        // Trending requires server-side sort (no score in local data).
                        self.trigger_search();
                    } else {
                        // Downloads/Likes/Recent can be sorted client-side.
                        self.apply_filters();
                    }
                }
            }
            PickerMode::Task => {
                if let Some(&val) = TaskFilter::ALL.get(self.picker_cursor) {
                    self.task_filter = val;
                    self.trigger_search();
                }
            }
            PickerMode::Library => {
                if let Some(&val) = LibraryFilter::ALL.get(self.picker_cursor) {
                    self.library_filter = val;
                    self.trigger_search();
                }
            }
        }
    }

    /// Confirm the picker selection and close the picker.
    fn confirm_picker(&mut self, mode: PickerMode) {
        self.apply_picker(mode);
        self.picker = None;
    }

    pub(super) fn run(
        &mut self,
        terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    ) -> Result<Option<String>> {
        self.trigger_search();
        self.apply_pinned_filters();
        self.reset_selection();

        loop {
            // Auto-dismiss error toast after 3 seconds.
            if let Some((_, t)) = &self.error_msg {
                if t.elapsed() > Duration::from_secs(3) {
                    self.error_msg = None;
                }
            }

            terminal.draw(|f| self.draw(f))?;

            // Check for search results
            while let Ok(resp) = self.rx.try_recv() {
                if resp.generation == self.generation {
                    self.pending = self.pending.saturating_sub(1);
                    if self.pending == 0 {
                        self.loading = false;
                    }
                    match resp.results {
                        Ok(results) => {
                            if resp.is_primary {
                                self.primary_arrived = true;
                                self.cached_query = self.query.clone();
                                self.results = results;
                                // Merge any secondaries that arrived first.
                                let stashed = std::mem::take(&mut self.stashed_secondary);
                                for secondary in stashed {
                                    self.merge_secondary(secondary);
                                }
                            } else if !self.primary_arrived {
                                // Secondary arrived before primary — stash it.
                                self.stashed_secondary.push(results);
                            } else {
                                self.merge_secondary(results);
                            }
                            self.error_msg = None;
                            self.apply_filters();
                        }
                        Err(e) => {
                            // Show error as transient toast; don't clear results.
                            self.error_msg = Some((e, Instant::now()));
                        }
                    }
                }
            }

            // Debounce
            if self.active_tab == TabMode::Search
                && self.query_dirty
                && self.last_keystroke.elapsed() > Duration::from_millis(300)
            {
                self.trigger_search();
            }

            // Non-blocking event poll
            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }

                    // Picker overlay active — handle picker keys.
                    if let Some(mode) = self.picker {
                        match key.code {
                            KeyCode::Char(c) if c.is_whitespace() => {
                                self.picker = None;
                                self.toggle_selected_pin();
                            }
                            KeyCode::Char(c) if c.is_ascii_digit() => {
                                let digit = (c as u8 - b'0') as usize;
                                self.handle_picker_digit(mode, digit);
                                self.picker = None;
                            }
                            KeyCode::Down => {
                                let count = mode.item_count();
                                if self.picker_cursor + 1 < count {
                                    self.picker_cursor += 1;
                                    self.apply_picker(mode);
                                }
                            }
                            KeyCode::Up => {
                                if self.picker_cursor > 0 {
                                    self.picker_cursor -= 1;
                                    self.apply_picker(mode);
                                }
                            }
                            KeyCode::Enter => {
                                self.confirm_picker(mode);
                            }
                            KeyCode::Esc => {
                                self.picker = None;
                            }
                            KeyCode::Right | KeyCode::Tab => {
                                self.open_picker(mode.next());
                            }
                            KeyCode::Left => {
                                self.open_picker(mode.prev());
                            }
                            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                return Ok(None);
                            }
                            _ => {
                                // Any other key dismisses picker.
                                self.picker = None;
                            }
                        }
                        continue;
                    }

                    match key.code {
                        KeyCode::Esc => {
                            if !self.query.is_empty() {
                                self.query.clear();
                                if self.active_tab == TabMode::Search {
                                    self.trigger_search();
                                } else {
                                    self.apply_pinned_filters();
                                    self.reset_selection();
                                }
                            } else {
                                return Ok(None);
                            }
                        }
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            return Ok(None);
                        }
                        KeyCode::Enter => {
                            if let Some(model_id) = self.current_selected_model_id() {
                                return Ok(Some(model_id));
                            }
                        }
                        KeyCode::Down => self.move_down(),
                        KeyCode::Up => self.move_up(),
                        KeyCode::Tab => {
                            self.open_picker(PickerMode::Task);
                        }
                        KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            self.active_tab = self.active_tab.toggle();
                            self.picker = None;
                            self.reset_selection();
                        }
                        KeyCode::Char('o')
                            if self.active_tab == TabMode::Search
                                && key.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            self.open_picker(PickerMode::Sort);
                        }
                        KeyCode::Char('t')
                            if self.active_tab == TabMode::Search
                                && key.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            self.open_picker(PickerMode::Task);
                        }
                        KeyCode::Char('s')
                            if self.active_tab == TabMode::Search
                                && key.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            self.open_picker(PickerMode::Size);
                        }
                        KeyCode::Char('d')
                            if self.active_tab == TabMode::Search
                                && key.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            self.open_picker(PickerMode::Date);
                        }
                        KeyCode::Char('l')
                            if self.active_tab == TabMode::Search
                                && key.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            self.open_picker(PickerMode::Library);
                        }
                        KeyCode::Char(c) if c.is_whitespace() => self.toggle_selected_pin(),
                        KeyCode::Backspace => {
                            if self.query.pop().is_some() {
                                if self.active_tab == TabMode::Search {
                                    self.last_keystroke = Instant::now();
                                    self.query_dirty = true;
                                    self.apply_filters();
                                } else {
                                    self.apply_pinned_filters();
                                    self.reset_selection();
                                }
                            }
                        }
                        KeyCode::Char(c) => {
                            self.query.push(c);
                            if self.active_tab == TabMode::Search {
                                self.last_keystroke = Instant::now();
                                self.query_dirty = true;
                                self.apply_filters();
                            } else {
                                self.apply_pinned_filters();
                                self.reset_selection();
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Handle a digit press inside a picker.
    fn handle_picker_digit(&mut self, mode: PickerMode, digit: usize) -> bool {
        match mode {
            PickerMode::Size => {
                let idx = if digit == 0 { 9 } else { digit - 1 };
                if let Some(&val) = SizeFilter::ALL.get(idx) {
                    self.size_filter = val;
                    self.apply_filters();
                    return true;
                }
            }
            PickerMode::Date => {
                let items = DateFilter::ALL;
                let idx = if digit == 0 {
                    items.len() - 1
                } else {
                    digit - 1
                };
                if let Some(&val) = items.get(idx) {
                    self.date_filter = val;
                    self.apply_filters();
                    return true;
                }
            }
            PickerMode::Sort => {
                if digit >= 1 && digit <= SortMode::ALL.len() {
                    self.sort_mode = SortMode::ALL[digit - 1];
                    self.trigger_search();
                    return true;
                }
            }
            PickerMode::Task => {
                let items = TaskFilter::ALL;
                let idx = if digit == 0 {
                    items.len() - 1
                } else {
                    digit - 1
                };
                if let Some(&val) = items.get(idx) {
                    self.task_filter = val;
                    self.trigger_search();
                    return true;
                }
            }
            PickerMode::Library => {
                let items = LibraryFilter::ALL;
                let idx = if digit == 0 {
                    items.len() - 1
                } else {
                    digit - 1
                };
                if let Some(&val) = items.get(idx) {
                    self.library_filter = val;
                    self.trigger_search();
                    return true;
                }
            }
        }
        false
    }

    fn current_items_count(&self) -> usize {
        match self.active_tab {
            TabMode::Search => self.filtered.len(),
            TabMode::Pinned => self.pinned_filtered.len(),
        }
    }

    fn model_id_at_index(&self, idx: usize) -> Option<String> {
        match self.active_tab {
            TabMode::Search => self.filtered.get(idx).map(|m| m.model_id.clone()),
            TabMode::Pinned => self.pinned_filtered.get(idx).cloned(),
        }
    }

    fn current_selected_model_id(&self) -> Option<String> {
        let idx = self.state.selected()?;
        self.model_id_at_index(idx)
    }

    fn toggle_selected_pin(&mut self) {
        let Some(model_id) = self.current_selected_model_id() else {
            return;
        };
        let selected_before = self.state.selected().unwrap_or(0);
        let outcome = if self.pinned_ids.contains(&model_id) {
            self.pin_store
                .unpin(&model_id)
                .map(|changed| (changed, false))
        } else {
            self.pin_store.pin(&model_id).map(|changed| (changed, true))
        };

        match outcome {
            Ok((changed, added)) => {
                if !changed {
                    return;
                }
                if added {
                    self.pinned_ids.insert(model_id.clone());
                    self.pinned_models.retain(|m| m != &model_id);
                    self.pinned_models.insert(0, model_id.clone());
                    let local_size = talu::repo::repo_size(&model_id);
                    if local_size > 0 {
                        let _ = self.pin_store.upsert_size_bytes(&model_id, local_size);
                    }
                } else {
                    self.pinned_ids.remove(&model_id);
                    self.pinned_models.retain(|m| m != &model_id);
                    let _ = self.pin_store.clear_size_bytes(&model_id);
                }
                self.apply_pinned_filters();
                let count = self.current_items_count();
                if count == 0 {
                    self.state.select(None);
                } else {
                    self.state.select(Some(selected_before.min(count - 1)));
                }
            }
            Err(e) => {
                self.error_msg = Some((e.to_string(), Instant::now()));
            }
        }
    }

    fn move_down(&mut self) {
        let count = self.current_items_count();
        if count == 0 {
            return;
        }
        let cur = self.state.selected().unwrap_or(0);
        if cur + 1 < count {
            self.state.select(Some(cur + 1));
        }
    }

    fn move_up(&mut self) {
        let count = self.current_items_count();
        if count == 0 {
            return;
        }
        let cur = self.state.selected().unwrap_or(0);
        if cur > 0 {
            self.state.select(Some(cur - 1));
        }
    }
}
