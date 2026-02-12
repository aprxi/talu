//! Interactive terminal model selector (ratatui).
//!
//! Shows all local models (managed first, then HF cache) with a search-as-you-type
//! filter. Typing narrows the list; arrow keys navigate; Enter selects.
//! Managed models are numbered — type a number to jump directly.

use std::io::{self, IsTerminal};

use anyhow::{bail, Result};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table, TableState};
use ratatui::Terminal;

use talu::CacheOrigin;

use crate::tui_common::{
    enter_tui, format_date, format_size, highlight_text, leave_tui, truncate_name,
};

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

struct ModelEntry {
    id: String,
    #[allow(dead_code)]
    source: CacheOrigin,
    size_str: String,
    date_str: String,
    type_str: String,
    quant_str: String,
}

/// Sentinel entry used as a visual separator between sections.
const SEPARATOR_IDX: usize = usize::MAX;
/// Sentinel for an empty spacer row.
const SPACER_IDX: usize = usize::MAX - 1;

/// Returns true for sentinel indices (separator, spacer) that are not real models.
fn is_sentinel(idx: usize) -> bool {
    idx == SEPARATOR_IDX || idx == SPACER_IDX
}

struct ModelSelector {
    /// All models, managed first then HF.
    all: Vec<ModelEntry>,
    /// Number of managed models (prefix of `all`).
    managed_count: usize,
    /// Indices into `all` matching the current query, with sentinel entries
    /// (SPACER_IDX, SEPARATOR_IDX) inserted between managed and HF sections.
    filtered: Vec<usize>,
    query: String,
    state: TableState,
    /// Last model saved via speed dial (shown as confirmation).
    last_saved: Option<String>,
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

fn get_model_info_for_display(cache_path: &str) -> (String, String) {
    if cache_path.is_empty() {
        return ("-".into(), "-".into());
    }
    let info = match talu::model::describe(cache_path) {
        Ok(info) => info,
        Err(_) => return ("-".into(), "-".into()),
    };
    let model_type = if info.model_type.is_empty() {
        "-".into()
    } else {
        normalize_model_type(&info.model_type)
    };
    let quant = format_quant(info.quant_method, info.quant_bits, info.quant_group_size);
    (model_type, quant)
}

fn normalize_model_type(model_type: &str) -> String {
    match model_type.to_lowercase().as_str() {
        "llama" => "Llama",
        "qwen2" | "qwen2_5" => "Qwen2",
        "qwen3" => "Qwen3",
        "qwen3_moe" => "Qwen3MoE",
        "gemma" => "Gemma",
        "gemma2" => "Gemma2",
        "gemma3" | "gemma3_text" => "Gemma3",
        "phi" | "phi3" | "phi4" => "Phi",
        "mistral" => "Mistral",
        "mistral3" => "Mistral3",
        "mixtral" => "Mixtral",
        "granite" => "Granite",
        "granitemoehybrid" => "GraniteMoE",
        "bert" => "BERT",
        "mpnet" => "MPNet",
        "gpt_oss" => "GPT-OSS",
        "smollm3" => "SmolLM3",
        "lfm2" => "LFM2",
        "deepseek_v3" => "DeepSeek",
        "kimi_vl" => "Kimi-VL",
        _ => return capitalize_first(model_type),
    }
    .to_string()
}

fn capitalize_first(s: &str) -> String {
    let mut chars: Vec<char> = s.chars().collect();
    if !chars.is_empty() {
        chars[0] = chars[0].to_ascii_uppercase();
    }
    chars.into_iter().collect()
}

fn format_quant(method: talu::QuantMethod, bits: i32, group_size: i32) -> String {
    match method {
        talu::QuantMethod::None => "F16".into(),
        talu::QuantMethod::Gaffine | talu::QuantMethod::Native => {
            format!("GAF{}_{}", bits, group_size)
        }
        talu::QuantMethod::Mxfp4 => "MXFP4".into(),
    }
}

// ---------------------------------------------------------------------------
// ModelSelector
// ---------------------------------------------------------------------------

impl ModelSelector {
    fn new() -> Result<Self> {
        let list = talu::repo::repo_list_models(false).map_err(|e| anyhow::anyhow!("{}", e))?;

        let mut managed = Vec::new();
        let mut hf = Vec::new();

        for m in &list {
            let size = talu::repo::repo_size(&m.id);
            let mtime = talu::repo::repo_mtime(&m.id);
            let (type_str, quant_str) = get_model_info_for_display(&m.path);

            let entry = ModelEntry {
                id: m.id.clone(),
                source: m.source,
                size_str: format_size(size),
                date_str: format_date(mtime),
                type_str,
                quant_str,
            };

            match m.source {
                CacheOrigin::Managed => managed.push(entry),
                CacheOrigin::Hub => hf.push(entry),
            }
        }

        let managed_count = managed.len();

        // Managed first, then HF
        let mut all = managed;
        all.extend(hf);

        let mut sel = Self {
            all,
            managed_count,
            filtered: Vec::new(),
            query: String::new(),
            state: TableState::default(),
            last_saved: None,
        };
        sel.refilter();
        Ok(sel)
    }

    /// Build the filtered list with separator between managed and HF sections.
    fn refilter(&mut self) {
        let q = self.query.to_lowercase();
        self.build_filtered_with_separator(|e| {
            if q.is_empty() {
                return true;
            }
            e.id.to_lowercase().contains(&q)
                || e.type_str.to_lowercase().contains(&q)
                || e.quant_str.to_lowercase().contains(&q)
        });

        // Select first non-sentinel
        let first = self.filtered.iter().position(|&i| !is_sentinel(i));
        self.state.select(first);
    }

    fn build_filtered_with_separator(&mut self, predicate: impl Fn(&ModelEntry) -> bool) {
        let mut managed_matches: Vec<usize> = Vec::new();
        let mut hf_matches: Vec<usize> = Vec::new();

        for (i, e) in self.all.iter().enumerate() {
            if !predicate(e) {
                continue;
            }
            if i < self.managed_count {
                managed_matches.push(i);
            } else {
                hf_matches.push(i);
            }
        }

        self.filtered = Vec::with_capacity(managed_matches.len() + hf_matches.len() + 3);

        if !managed_matches.is_empty() {
            self.filtered.extend(&managed_matches);
        }

        // No separator — managed and HF flow as one numbered list

        if !hf_matches.is_empty() {
            self.filtered.extend(&hf_matches);
        }
    }

    fn selection(&self) -> Option<&ModelEntry> {
        self.state
            .selected()
            .and_then(|i| self.filtered.get(i))
            .filter(|&&idx| !is_sentinel(idx))
            .map(|&idx| &self.all[idx])
    }

    /// Get the filtered-list position for the Nth visible (non-sentinel) result (1-based).
    fn nth_visible(&self, n: usize) -> Option<usize> {
        let mut count = 0usize;
        for (pos, &idx) in self.filtered.iter().enumerate() {
            if is_sentinel(idx) {
                continue;
            }
            count += 1;
            if count == n {
                return Some(pos);
            }
        }
        None
    }

    /// Count real (non-separator) entries in filtered list.
    fn match_count(&self) -> usize {
        self.filtered.iter().filter(|&&i| !is_sentinel(i)).count()
    }

    fn move_down(&mut self) {
        let len = self.filtered.len();
        if len == 0 {
            return;
        }
        let cur = self.state.selected().unwrap_or(0);
        // Skip sentinel rows
        let mut next = cur + 1;
        while next < len && is_sentinel(self.filtered[next]) {
            next += 1;
        }
        if next < len {
            self.state.select(Some(next));
        }
    }

    fn move_up(&mut self) {
        if self.filtered.is_empty() {
            return;
        }
        let cur = self.state.selected().unwrap_or(0);
        if cur == 0 {
            return;
        }
        // Skip sentinel rows
        let mut prev = cur - 1;
        while prev > 0 && is_sentinel(self.filtered[prev]) {
            prev -= 1;
        }
        if !is_sentinel(self.filtered[prev]) {
            self.state.select(Some(prev));
        }
    }

    fn run(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    ) -> Result<Option<String>> {
        loop {
            terminal.draw(|f| self.draw(f))?;

            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Esc => {
                        if !self.query.is_empty() {
                            self.query.clear();
                            self.refilter();
                        } else {
                            return Ok(self.last_saved.clone());
                        }
                    }
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        return Ok(self.last_saved.clone());
                    }
                    KeyCode::Enter => {
                        if let Some(sel) = self.selection() {
                            let id = sel.id.clone();
                            crate::config::set_default_model(&id)?;
                            return Ok(Some(id));
                        }
                    }
                    KeyCode::Down => self.move_down(),
                    KeyCode::Up => self.move_up(),
                    KeyCode::Backspace => {
                        self.query.pop();
                        self.refilter();
                    }
                    KeyCode::Char(c) if c.is_ascii_digit() => {
                        // Digits are speed dial: pick Nth visible result (1-9)
                        // Press same number twice to confirm and exit.
                        let n: usize = (c as u8 - b'0') as usize;
                        if n >= 1 {
                            if let Some(pos) = self.nth_visible(n) {
                                let orig_idx = self.filtered[pos];
                                let id = self.all[orig_idx].id.clone();
                                if self.last_saved.as_ref() == Some(&id) {
                                    return Ok(Some(id));
                                }
                                crate::config::set_default_model(&id)?;
                                self.last_saved = Some(id);
                                self.state.select(Some(pos));
                            }
                        }
                    }
                    KeyCode::Char(c) => {
                        self.query.push(c);
                        self.refilter();
                    }
                    _ => {}
                }
            }
        }
    }

    fn draw(&mut self, f: &mut ratatui::Frame) {
        let area = f.area();

        let chunks = Layout::vertical([
            Constraint::Length(5), // search box (bigger)
            Constraint::Min(4),    // model table
            Constraint::Length(1), // footer
        ])
        .split(area);

        self.draw_search(f, chunks[0]);
        self.draw_table(f, chunks[1]);
        self.draw_footer(f, chunks[2]);
    }

    fn draw_search(&self, f: &mut ratatui::Frame, area: Rect) {
        let prompt_style = Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD);
        let query_style = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        let cursor_style = Style::default().fg(Color::Yellow);
        let placeholder_style = Style::default().fg(Color::DarkGray);

        let input_line = if self.query.is_empty() {
            Line::from(vec![
                Span::styled(" > ", prompt_style),
                Span::styled("type to search, then pick 1-9...", placeholder_style),
            ])
        } else {
            Line::from(vec![
                Span::styled(" > ", prompt_style),
                Span::styled(&self.query, query_style),
                Span::styled("█", cursor_style),
            ])
        };

        // Status line below the input
        let matches = self.match_count();
        let total = self.all.len();
        let status_line = if let Some(ref saved) = self.last_saved {
            Line::from(vec![
                Span::styled(
                    "  ✓ ",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(saved.clone(), Style::default().fg(Color::Green)),
            ])
        } else if !self.query.is_empty() {
            Line::styled(
                format!("  {} of {} models", matches, total),
                Style::default().fg(Color::DarkGray),
            )
        } else {
            Line::raw("")
        };

        let title = format!(" Select Default Model ({} models) ", total);

        let block = Block::default()
            .title(title)
            .title_style(
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));

        let paragraph = Paragraph::new(vec![Line::raw(""), input_line, status_line]).block(block);
        f.render_widget(paragraph, area);
    }

    fn draw_table(&mut self, f: &mut ratatui::Frame, area: Rect) {
        if self.filtered.is_empty() || self.match_count() == 0 {
            let msg = if self.query.is_empty() {
                "(no models found — run 'talu get Org/Model' to download)"
            } else {
                "(no matches)"
            };
            let paragraph =
                Paragraph::new(format!("  {}", msg)).style(Style::default().fg(Color::DarkGray));
            f.render_widget(paragraph, area);
            return;
        }

        let search_query = &self.query;

        let dim = Style::default().fg(Color::DarkGray);
        let num_style = Style::default().fg(Color::White);
        let sep_style = Style::default().fg(Color::DarkGray);

        let mut visible_num = 0usize;

        let rows: Vec<Row> = self
            .filtered
            .iter()
            .map(|&orig_idx| {
                if orig_idx == SPACER_IDX {
                    return Row::new(vec![
                        Line::raw(""),
                        Line::raw(""),
                        Line::raw(""),
                        Line::raw(""),
                        Line::raw(""),
                    ]);
                }
                if orig_idx == SEPARATOR_IDX {
                    return Row::new(vec![
                        Line::raw(""),
                        Line::styled("── HuggingFace cache ──", sep_style),
                        Line::raw(""),
                        Line::raw(""),
                        Line::raw(""),
                    ]);
                }

                visible_num += 1;
                let e = &self.all[orig_idx];
                let display_name = truncate_name(&e.id, 60);
                let model_line = highlight_text(&display_name, search_query, Style::default());

                // Number the first 9 visible results for speed dial
                let num_cell = if visible_num <= 9 {
                    Line::styled(format!("{:>3} ", visible_num), num_style)
                } else {
                    Line::raw("")
                };

                Row::new(vec![
                    num_cell,
                    model_line,
                    Line::styled(e.size_str.clone(), dim),
                    Line::styled(format!("{}  {}", e.type_str, e.quant_str), dim),
                    Line::styled(e.date_str.clone(), dim),
                ])
            })
            .collect();

        let widths = [
            Constraint::Length(4),  // #
            Constraint::Length(60), // MODEL (capped at 60 chars)
            Constraint::Length(8),  // SIZE
            Constraint::Length(18), // TYPE + QUANT
            Constraint::Length(14), // DATE
        ];

        let table = Table::new(rows, widths)
            .row_highlight_style(
                Style::default()
                    .add_modifier(Modifier::REVERSED)
                    .fg(Color::White),
            )
            .highlight_symbol("▸")
            .block(Block::default());

        f.render_stateful_widget(table, area, &mut self.state);
    }

    fn draw_footer(&self, f: &mut ratatui::Frame, area: Rect) {
        let key = Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);
        let dim = Style::default().fg(Color::DarkGray);

        let footer = Line::from(vec![
            Span::raw(" "),
            Span::styled("1-9", key),
            Span::styled(" Pick  ", dim),
            Span::styled("↑↓", key),
            Span::styled(" Navigate  ", dim),
            Span::styled("Enter", key),
            Span::styled(" Select  ", dim),
            Span::styled("Esc", key),
            Span::styled(
                if self.query.is_empty() {
                    " Quit"
                } else {
                    " Clear"
                },
                dim,
            ),
        ]);
        f.render_widget(Paragraph::new(footer), area);
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the interactive model selector. Returns `Some(model_id)` on selection,
/// `None` on cancel, or an error if no TTY is available.
pub fn run_interactive_selector() -> Result<Option<String>> {
    if !io::stdout().is_terminal() {
        bail!("Interactive selector requires a terminal. Use -m/--model flag or set MODEL_URI.");
    }

    let mut terminal = enter_tui()?;
    let mut selector = ModelSelector::new()?;
    let result = selector.run(&mut terminal);
    leave_tui(&mut terminal);

    result
}
