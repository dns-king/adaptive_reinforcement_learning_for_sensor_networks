//! Dependency-free local SVG outputs, deliberately suitable for offline experiment reports.
use crate::experiments::EpisodeMetrics;
use std::{fs, path::Path};
fn svg(w: u32, h: u32, body: String) -> String {
    format!("<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'><rect width='100%' height='100%' fill='white'/>{body}</svg>")
}
pub fn learning_curve(path: &Path, xs: &[EpisodeMetrics]) -> std::io::Result<()> {
    let w = 900.;
    let h = 420.;
    let lo = xs.iter().map(|x| x.reward).fold(f64::INFINITY, f64::min);
    let hi = xs
        .iter()
        .map(|x| x.reward)
        .fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(1e-9);
    let points = xs
        .iter()
        .enumerate()
        .map(|(i, x)| {
            format!(
                "{:.1},{:.1}",
                50. + i as f64 * (w - 70.) / (xs.len().saturating_sub(1).max(1) as f64),
                h - 35. - (x.reward - lo) / span * (h - 70.)
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(path,svg(900,420,format!("<text x='45' y='25' font-family='sans-serif'>Learning curve: episodic reward</text><path d='M50 20V385H880' stroke='#555' fill='none'/><polyline points='{points}' fill='none' stroke='#1769aa' stroke-width='2'/><text x='50' y='410'>episode</text><text x='5' y='50'>reward</text>")))
}
pub fn topology(path: &Path, health: &[f64]) -> std::io::Result<()> {
    let (w, h) = (600., 600.);
    let mut b=String::from("<text x='20' y='30' font-family='sans-serif'>Network topology — fill indicates sensor health</text><circle cx='300' cy='310' r='20' fill='#333'/>");
    for (i, x) in health.iter().enumerate() {
        let a = i as f64 / health.len() as f64 * std::f64::consts::TAU;
        let (cx, cy) = (300. + 220. * a.cos(), 310. + 220. * a.sin());
        let c = format!("rgb({},{},70)", ((1. - x) * 220.) as u8, (x * 180.) as u8);
        b.push_str(&format!("<line x1='300' y1='310' x2='{cx:.0}' y2='{cy:.0}' stroke='#bbb'/><circle cx='{cx:.0}' cy='{cy:.0}' r='16' fill='{c}'/><text x='{cx:.0}' y='{y:.0}' font-size='10'>{i}</text>",y=cy+4.));
    }
    fs::write(path, svg(w as u32, h as u32, b))
}
