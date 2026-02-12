//! Draw methods for HfSearchSelector (ratatui rendering).

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Row, Table};

use crate::tui_common::{highlight_text, truncate_name};

use super::filters::*;
use super::selector::{HfSearchSelector, PickerMode, TabMode};
use super::{format_date_iso, format_downloads, format_params, LabelPositions};

impl HfSearchSelector {
    pub(super) fn draw(&mut self, f: &mut ratatui::Frame) {
        let area = f.area();
        let chunks = Layout::vertical([
            Constraint::Length(5), // search box
            Constraint::Min(4),    // results table
            Constraint::Length(1), // footer
        ])
        .split(area);

        self.draw_search(f, chunks[0]);
        self.draw_table(f, chunks[1]);
        self.draw_footer(f, chunks[2]);

        // Draw dropdown overlay on top of everything, anchored to label.
        if self.active_tab == TabMode::Search {
            if let Some(mode) = self.picker {
                self.draw_dropdown(f, chunks[0], chunks[1], mode);
            }
        }

        // Draw error toast over the footer bar (full width, solid background).
        if let Some((ref msg, _)) = self.error_msg {
            let err_text = format!(" error: {} ", msg);
            let rect = Rect::new(area.x, chunks[2].y, area.width, 1);
            f.render_widget(Clear, rect);
            f.render_widget(
                Paragraph::new(err_text).style(
                    Style::default()
                        .fg(Color::White)
                        .bg(Color::Red)
                        .add_modifier(Modifier::BOLD),
                ),
                rect,
            );
        }
    }

    /// Render a floating dropdown anchored below the label for `mode`.
    fn draw_dropdown(
        &self,
        f: &mut ratatui::Frame,
        search_area: Rect,
        table_area: Rect,
        mode: PickerMode,
    ) {
        let dim = Style::default().fg(Color::DarkGray);
        let key_style = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        let active_style = Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD);
        let label_style = Style::default().fg(Color::White);
        let cursor_style = Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::REVERSED);

        struct Item {
            key: u8,
            label: &'static str,
            active: bool,
        }

        let items: Vec<Item> = match mode {
            PickerMode::Size => SizeFilter::ALL
                .iter()
                .enumerate()
                .map(|(i, &v)| Item {
                    key: if i < 9 { (i + 1) as u8 } else { 0 },
                    label: v.label(),
                    active: v == self.size_filter,
                })
                .collect(),
            PickerMode::Date => DateFilter::ALL
                .iter()
                .enumerate()
                .map(|(i, &v)| Item {
                    key: if i < DateFilter::ALL.len() - 1 {
                        (i + 1) as u8
                    } else {
                        0
                    },
                    label: v.label(),
                    active: v == self.date_filter,
                })
                .collect(),
            PickerMode::Sort => SortMode::ALL
                .iter()
                .enumerate()
                .map(|(i, &v)| Item {
                    key: (i + 1) as u8,
                    label: v.label(),
                    active: v == self.sort_mode,
                })
                .collect(),
            PickerMode::Task => TaskFilter::ALL
                .iter()
                .enumerate()
                .map(|(i, &v)| Item {
                    key: if i < TaskFilter::ALL.len() - 1 {
                        (i + 1) as u8
                    } else {
                        0
                    },
                    label: v.label(),
                    active: v == self.task_filter,
                })
                .collect(),
            PickerMode::Library => LibraryFilter::ALL
                .iter()
                .enumerate()
                .map(|(i, &v)| Item {
                    key: if i < LibraryFilter::ALL.len() - 1 {
                        (i + 1) as u8
                    } else {
                        0
                    },
                    label: v.label(),
                    active: v == self.library_filter,
                })
                .collect(),
        };

        // Build lines for the dropdown.
        let mut lines = Vec::new();
        for (i, item) in items.iter().enumerate() {
            let is_cursor = i == self.picker_cursor;
            let style = if is_cursor {
                cursor_style
            } else if item.active {
                active_style
            } else {
                label_style
            };
            let marker = if item.active { " \u{25cf}" } else { "" };
            let pointer = if is_cursor { "\u{25b8}" } else { " " };
            lines.push(Line::from(vec![
                Span::styled(
                    format!(" {}", pointer),
                    if is_cursor { active_style } else { dim },
                ),
                Span::styled(format!("{}", item.key), key_style),
                Span::styled(format!(" {}{} ", item.label, marker), style),
            ]));
        }

        // Calculate dropdown width from longest line content.
        let max_label_w = items
            .iter()
            .map(|it| {
                // " ▸" + "N" + " label ●" + " " padding
                3 + 1 + 1 + it.label.len() + if it.active { 2 } else { 0 } + 1
            })
            .max()
            .unwrap_or(10) as u16;
        // Add 2 for border.
        let dropdown_w = max_label_w + 2;
        let dropdown_h = items.len() as u16 + 2; // +2 for borders

        // Anchor x from the label position.
        let label_x = self.label_pos.x_for(mode);
        // The search box has a 1-cell border on the left, and the status line
        // is inside the box.  search_area.x is the outer left.  The inner
        // content starts at search_area.x + 1.
        let anchor_x = search_area.x + 1 + label_x;
        // Clamp so we don't overflow the right edge.
        let screen_w = f.area().width;
        let x = if anchor_x + dropdown_w > screen_w {
            screen_w.saturating_sub(dropdown_w)
        } else {
            anchor_x
        };
        // The dropdown appears just below the search box border.
        let y = search_area.y + search_area.height;
        // Clamp height to available space.
        let max_h = table_area.height.min(dropdown_h);

        let rect = Rect::new(x, y, dropdown_w, max_h);

        // Clear the area behind the dropdown, then render.
        f.render_widget(Clear, rect);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));
        let paragraph = Paragraph::new(lines).block(block);
        f.render_widget(paragraph, rect);
    }

    fn draw_search(&mut self, f: &mut ratatui::Frame, area: Rect) {
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
                Span::styled("type to search HuggingFace models...", placeholder_style),
            ])
        } else {
            Line::from(vec![
                Span::styled(" > ", prompt_style),
                Span::styled(&self.query, query_style),
                Span::styled("\u{2588}", cursor_style),
            ])
        };

        let tag_style = Style::default().fg(Color::Cyan);
        let dim = Style::default().fg(Color::DarkGray);
        let hi = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        // Style for the label that is currently open as a dropdown.
        let open_style = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED);

        let status_line = if let Some(ref saved) = self.last_saved {
            self.label_pos = LabelPositions::default();
            Line::from(vec![
                Span::styled(
                    "  \u{2713} ",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(saved.clone(), Style::default().fg(Color::Green)),
                Span::styled("  (press again to download)", dim),
            ])
        } else if self.active_tab == TabMode::Search && self.loading {
            // Keep label_pos from last render so dropdown stays anchored.
            Line::styled("  searching...", dim)
        } else if self.active_tab == TabMode::Pinned {
            self.label_pos = LabelPositions::default();
            Line::from(vec![
                Span::styled(
                    format!("  {} pinned", self.pinned_filtered.len()),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled(
                    format!("  ({} total)", self.pinned_models.len()),
                    Style::default().fg(Color::DarkGray),
                ),
            ])
        } else {
            // Build status line and track x-offsets for each label.
            let count_text = format!("  {} results", self.filtered.len());

            let open = self.picker;

            // Track cumulative width to record label positions.
            let mut col: u16 = 0;
            let mut spans: Vec<Span> = Vec::new();

            // Helper: push a span and advance col by display width (char count).
            macro_rules! push {
                ($text:expr, $style:expr) => {{
                    let t: String = $text;
                    col += t.chars().count() as u16;
                    spans.push(Span::styled(t, $style));
                }};
            }

            push!(count_text, dim);
            push!("  ".into(), dim);

            // Task label
            let task_open = open == Some(PickerMode::Task);
            self.label_pos.task_x = col;
            if task_open {
                push!("Task".into(), open_style);
                push!("\u{25be}".into(), open_style); // ▾ down-pointing triangle
            } else {
                push!("T".into(), hi);
                push!("ask".into(), dim);
            }
            push!(":".into(), dim);
            push!(self.task_filter.label().into(), tag_style);
            push!("  ".into(), dim);

            // Size label
            let size_open = open == Some(PickerMode::Size);
            self.label_pos.size_x = col;
            if size_open {
                push!("Size".into(), open_style);
                push!("\u{25be}".into(), open_style);
            } else {
                push!("S".into(), hi);
                push!("ize".into(), dim);
            }
            push!(":".into(), dim);
            push!(self.size_filter.label().into(), tag_style);
            push!("  ".into(), dim);

            // Date label
            let date_open = open == Some(PickerMode::Date);
            self.label_pos.date_x = col;
            if date_open {
                push!("Date".into(), open_style);
                push!("\u{25be}".into(), open_style);
            } else {
                push!("D".into(), hi);
                push!("ate".into(), dim);
            }
            push!(":".into(), dim);
            push!(self.date_filter.label().into(), tag_style);
            push!("  ".into(), dim);

            // Library label
            let lib_open = open == Some(PickerMode::Library);
            self.label_pos.lib_x = col;
            if lib_open {
                push!("Lib".into(), open_style);
                push!("\u{25be}".into(), open_style);
            } else {
                push!("L".into(), hi);
                push!("ib".into(), dim);
            }
            push!(":".into(), dim);
            push!(self.library_filter.label().into(), tag_style);
            push!("  ".into(), dim);

            // Sort label
            let sort_open = open == Some(PickerMode::Sort);
            self.label_pos.sort_x = col;
            if sort_open {
                push!("Sort".into(), open_style);
                push!("\u{25be}".into(), open_style);
            } else {
                push!("s".into(), dim);
                push!("O".into(), hi);
                push!("rt".into(), dim);
            }
            push!(":".into(), dim);
            push!(self.sort_mode.label().into(), tag_style);
            let _ = col;

            Line::from(spans)
        };

        let title = match self.active_tab {
            TabMode::Search => " Search HuggingFace Models  [Search* | Pinned] ",
            TabMode::Pinned => " Search HuggingFace Models  [Search | Pinned*] ",
        };
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
        let dim = Style::default().fg(Color::DarkGray);
        let cached_row = Style::default().bg(Color::Rgb(20, 35, 20));

        if self.active_tab == TabMode::Pinned {
            if self.pinned_filtered.is_empty() {
                let msg = if self.pinned_models.is_empty() {
                    "(no pinned models yet)"
                } else if self.query.is_empty() {
                    "(no pinned models)"
                } else {
                    "(no pinned matches)"
                };
                let paragraph = Paragraph::new(format!("  {}", msg))
                    .style(Style::default().fg(Color::DarkGray));
                f.render_widget(paragraph, area);
                return;
            }

            let search_query = &self.query;
            let rows: Vec<Row> = self
                .pinned_filtered
                .iter()
                .map(|model_id| {
                    let is_cached = self.cached_ids.contains(model_id);
                    let display_name = truncate_name(model_id, 60);
                    let model_line = Line::from(
                        highlight_text(&display_name, search_query, Style::default()).spans,
                    );

                    let pin_cell = Line::styled("\u{2605}", Style::default().fg(Color::Yellow));
                    let row_style = if is_cached {
                        cached_row
                    } else {
                        Style::default()
                    };

                    Row::new(vec![pin_cell, model_line]).style(row_style)
                })
                .collect();

            let widths = [
                Constraint::Length(2),  // P
                Constraint::Length(62), // MODEL
            ];

            let header = Row::new(vec![Line::styled("P", dim), Line::styled("MODEL", dim)]);

            let table = Table::new(rows, widths)
                .header(header)
                .column_spacing(0)
                .row_highlight_style(
                    Style::default()
                        .add_modifier(Modifier::REVERSED)
                        .fg(Color::White),
                )
                .highlight_symbol("\u{25b8}")
                .block(Block::default());

            f.render_stateful_widget(table, area, &mut self.state);
            return;
        }

        if self.filtered.is_empty() {
            let msg = if self.loading {
                "(loading...)"
            } else if !self.results.is_empty() {
                "(all results filtered out)"
            } else if self.query.is_empty() {
                "(no models found)"
            } else {
                "(no matches)"
            };
            let paragraph =
                Paragraph::new(format!("  {}", msg)).style(Style::default().fg(Color::DarkGray));
            f.render_widget(paragraph, area);
            return;
        }

        let search_query = &self.query;
        let pinned_style = Style::default().fg(Color::Yellow);

        let rows: Vec<Row> = self
            .filtered
            .iter()
            .map(|r| {
                let is_cached = self.cached_ids.contains(&r.model_id);
                let is_pinned = self.pinned_ids.contains(&r.model_id);
                let name_max = 48;
                let display_name = truncate_name(&r.model_id, name_max);
                let model_line =
                    Line::from(highlight_text(&display_name, search_query, Style::default()).spans);
                let pin_cell = if is_pinned {
                    Line::styled("\u{2605}", pinned_style)
                } else {
                    Line::styled(" ", dim)
                };
                let row_style = if is_cached {
                    cached_row
                } else {
                    Style::default()
                };

                Row::new(vec![
                    pin_cell,
                    model_line,
                    Line::styled(format!("{:>6}", format_params(r.params_total)), dim),
                    Line::styled(format!("{:>8}", format_downloads(r.downloads)), dim),
                    Line::styled(format!("{:>6}", format_downloads(r.likes)), dim),
                    Line::styled(format_date_iso(&r.last_modified), dim),
                ])
                .style(row_style)
            })
            .collect();

        let widths = [
            Constraint::Length(2),  // P
            Constraint::Length(48), // MODEL
            Constraint::Length(7),  // SIZE
            Constraint::Length(9),  // DOWNLOADS
            Constraint::Length(7),  // LIKES
            Constraint::Length(12), // UPDATED
        ];

        let header = Row::new(vec![
            Line::styled("P", dim),
            Line::styled("MODEL", dim),
            Line::styled("  SIZE", dim),
            Line::styled("     DOWN", dim),
            Line::styled(" LIKES", dim),
            Line::styled("UPDATED", dim),
        ]);

        let table = Table::new(rows, widths)
            .header(header)
            .column_spacing(0)
            .row_highlight_style(
                Style::default()
                    .add_modifier(Modifier::REVERSED)
                    .fg(Color::White),
            )
            .highlight_symbol("\u{25b8}")
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
            Span::styled("\u{2191}\u{2193}", key),
            Span::styled(" Nav  ", dim),
            Span::styled("Enter", key),
            Span::styled(" Download  ", dim),
            Span::styled("Space", key),
            Span::styled(" Pin/Unpin  ", dim),
            Span::styled("Tab", key),
            Span::styled(" Filters  ", dim),
            Span::styled("Ctrl+P", key),
            Span::styled(" Search/Pinned  ", dim),
            Span::styled("Ctrl+T/S/D/L/O", key),
            Span::styled(" Filters  ", dim),
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
