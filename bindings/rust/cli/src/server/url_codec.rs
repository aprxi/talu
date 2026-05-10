fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn decode_component(raw: &str, plus_as_space: bool) -> String {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                if let (Some(hi), Some(lo)) = (hex_value(bytes[i + 1]), hex_value(bytes[i + 2])) {
                    out.push((hi << 4) | lo);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b'+' if plus_as_space => {
                out.push(b' ');
                i += 1;
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub(crate) fn decode_path_component(raw: &str) -> String {
    decode_component(raw, false)
}

pub(crate) fn parse_query_pairs(query: Option<&str>) -> Vec<(String, String)> {
    let Some(query) = query else {
        return Vec::new();
    };
    query
        .split('&')
        .filter(|segment| !segment.is_empty())
        .map(|segment| {
            let (key, value) = segment.split_once('=').unwrap_or((segment, ""));
            (decode_component(key, true), decode_component(value, true))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_path_component_decodes_percent_without_plus_space() {
        assert_eq!(
            decode_path_component("meta-llama%2FLlama+3%2E2"),
            "meta-llama/Llama+3.2"
        );
    }

    #[test]
    fn parse_query_pairs_decodes_form_encoding() {
        let pairs = parse_query_pairs(Some("query=hello+world&token=hf%2Fabc&flag"));
        assert_eq!(
            pairs,
            vec![
                ("query".to_string(), "hello world".to_string()),
                ("token".to_string(), "hf/abc".to_string()),
                ("flag".to_string(), "".to_string()),
            ]
        );
    }
}
