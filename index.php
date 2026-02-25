<?php
/**
 * Cholera Risk Assessment - Monte Carlo Simulation
 * PHP Implementation replicating Python app.py logic
 * 
 * Uses:
 *  - Poisson process for outbreak probability
 *  - Beta distribution for infection probability
 *  - Uniform distributions for false negative rates
 *  - Monte Carlo sampling for final risk estimation
 *  - Spearman rank correlation for sensitivity analysis
 */

// ============================================================
// STATISTICAL HELPER FUNCTIONS
// ============================================================

/**
 * Generate a random sample from Beta distribution using Gamma variates
 * Beta(a, b) = Gamma(a,1) / (Gamma(a,1) + Gamma(b,1))
 */
function gamma_variate($shape, $scale = 1.0) {
    // Marsaglia and Tsang's method for shape >= 1
    if ($shape < 1) {
        // For shape < 1, use Ahrens-Dieter method
        $u = mt_rand() / mt_getrandmax();
        return gamma_variate(1.0 + $shape, $scale) * pow($u, 1.0 / $shape);
    }
    
    $d = $shape - 1.0/3.0;
    $c = 1.0 / sqrt(9.0 * $d);
    
    while (true) {
        do {
            $x = random_normal();
            $v = 1.0 + $c * $x;
        } while ($v <= 0);
        
        $v = $v * $v * $v;
        $u = mt_rand() / mt_getrandmax();
        
        if ($u < 1.0 - 0.0331 * ($x * $x) * ($x * $x)) {
            return $d * $v * $scale;
        }
        
        if (log($u) < 0.5 * $x * $x + $d * (1.0 - $v + log($v))) {
            return $d * $v * $scale;
        }
    }
}

/**
 * Generate standard normal random number (Box-Muller transform)
 */
function random_normal($mean = 0.0, $stddev = 1.0) {
    static $spare = null;
    static $hasSpare = false;
    
    if ($hasSpare) {
        $hasSpare = false;
        $val = $spare;
        $spare = null;
        return $mean + $stddev * $val;
    }
    
    do {
        $u = (mt_rand() / mt_getrandmax()) * 2.0 - 1.0;
        $v = (mt_rand() / mt_getrandmax()) * 2.0 - 1.0;
        $s = $u * $u + $v * $v;
    } while ($s >= 1.0 || $s == 0.0);
    
    $s = sqrt(-2.0 * log($s) / $s);
    $spare = $v * $s;
    $hasSpare = true;
    
    return $mean + $stddev * $u * $s;
}

/**
 * Generate random sample from Beta distribution
 */
function random_beta($alpha, $beta_param) {
    $x = gamma_variate($alpha);
    $y = gamma_variate($beta_param);
    return $x / ($x + $y);
}

/**
 * Generate random sample from Uniform distribution
 */
function random_uniform($low, $high) {
    return $low + (mt_rand() / mt_getrandmax()) * ($high - $low);
}

/**
 * Spearman rank correlation coefficient
 */
function spearman_correlation($x, $y) {
    $n = count($x);
    if ($n < 2) return 0;
    
    // Rank the arrays
    $rank_x = compute_ranks($x);
    $rank_y = compute_ranks($y);
    
    // Compute Pearson correlation of ranks
    $mean_rx = array_sum($rank_x) / $n;
    $mean_ry = array_sum($rank_y) / $n;
    
    $num = 0; $den_x = 0; $den_y = 0;
    for ($i = 0; $i < $n; $i++) {
        $dx = $rank_x[$i] - $mean_rx;
        $dy = $rank_y[$i] - $mean_ry;
        $num += $dx * $dy;
        $den_x += $dx * $dx;
        $den_y += $dy * $dy;
    }
    
    $den = sqrt($den_x * $den_y);
    return ($den == 0) ? 0 : $num / $den;
}

/**
 * Compute ranks for an array (average rank for ties)
 */
function compute_ranks($arr) {
    $n = count($arr);
    $indexed = [];
    for ($i = 0; $i < $n; $i++) {
        $indexed[] = ['val' => $arr[$i], 'idx' => $i];
    }
    usort($indexed, function($a, $b) {
        return $a['val'] <=> $b['val'];
    });
    
    $ranks = array_fill(0, $n, 0);
    $i = 0;
    while ($i < $n) {
        $j = $i;
        while ($j < $n - 1 && $indexed[$j + 1]['val'] == $indexed[$i]['val']) {
            $j++;
        }
        $avg_rank = ($i + $j) / 2.0 + 1;
        for ($k = $i; $k <= $j; $k++) {
            $ranks[$indexed[$k]['idx']] = $avg_rank;
        }
        $i = $j + 1;
    }
    return $ranks;
}

/**
 * Compute percentile of an array
 */
function percentile($arr, $p) {
    sort($arr);
    $n = count($arr);
    $idx = ($p / 100.0) * ($n - 1);
    $lower = (int)floor($idx);
    $upper = (int)ceil($idx);
    if ($lower == $upper) return $arr[$lower];
    $frac = $idx - $lower;
    return $arr[$lower] * (1 - $frac) + $arr[$upper] * $frac;
}


// ============================================================
// CHOLERA RISK ASSESSMENT MODEL
// ============================================================

function cholera_risk_assessment($n_samples = 10000, $seed = 42) {
    mt_srand($seed);
    
    // Given Parameters
    $t = 1;
    $lambda_rate = 2779 / (22 * 12); // From table
    
    // Step 1: Probability of Cholera outbreak in Bangladesh (Poisson process)
    $P1 = 1 - exp(-$t * $lambda_rate);
    
    // Monte Carlo arrays
    $P2 = []; $P3 = []; $P4 = [];
    $P_final_samples = [];
    
    // Step 5: Exposure risk due to unsafe water and sanitation access
    $P5a = 0.11;
    $P5b = 0.01;
    $P5 = $P5a * $P5b;
    
    // Step 6: Probability of mortality
    $P6 = 0.03;
    
    for ($i = 0; $i < $n_samples; $i++) {
        // Step 2: Probability of Cholera infection in humans (Beta distribution)
        $p2_val = random_beta(1605, 24618);
        $P2[] = $p2_val;
        
        // Step 3: False Negative clinical examination in Bangladesh
        $p3a = random_uniform(0.549, 0.906);
        $p3_val = 1 - $p3a;
        $P3[] = $p3_val;
        
        // Step 4: FN clinical examination in India
        $p4a = random_uniform(0.884, 0.999);
        $p4_val = 1 - $p4a;
        $P4[] = $p4_val;
        
        // Final probability
        $P_final_samples[] = $P1 * $p2_val * $p3_val * $p4_val * $P5 * $P6;
    }
    
    // Statistics
    $expected_prob = array_sum($P_final_samples) / $n_samples;
    $ci_5 = percentile($P_final_samples, 5.0);
    $ci_95 = percentile($P_final_samples, 95.0);
    $min_val = min($P_final_samples);
    $max_val = max($P_final_samples);
    
    return [
        'P_final_samples' => $P_final_samples,
        'expected_prob'    => $expected_prob,
        'ci'               => [$ci_5, $ci_95],
        'min_val'          => $min_val,
        'max_val'          => $max_val,
        'P1'               => $P1,
        'P2'               => $P2,
        'P3'               => $P3,
        'P4'               => $P4,
        'P5'               => $P5,
        'P6'               => $P6,
        'lambda_rate'      => $lambda_rate,
        'n_samples'        => $n_samples,
    ];
}

// ============================================================
// SENSITIVITY ANALYSIS
// ============================================================

function sensitivity_analysis($P_final_samples, $P2, $P3, $P4) {
    return [
        'P2 (Infection Rate)'     => spearman_correlation($P2, $P_final_samples),
        'P3 (FN Bangladesh)'      => spearman_correlation($P3, $P_final_samples),
        'P4 (FN India)'           => spearman_correlation($P4, $P_final_samples),
    ];
}

// ============================================================
// BUILD HISTOGRAM DATA (for Chart.js)
// ============================================================

function build_histogram($samples, $bins = 50) {
    $min_s = min($samples);
    $max_s = max($samples);
    $range = $max_s - $min_s;
    if ($range == 0) $range = 1;
    $bin_width = $range / $bins;
    
    $counts = array_fill(0, $bins, 0);
    $labels = [];
    
    for ($i = 0; $i < $bins; $i++) {
        $labels[] = sprintf("%.2e", $min_s + $bin_width * ($i + 0.5));
    }
    
    foreach ($samples as $val) {
        $idx = (int)floor(($val - $min_s) / $bin_width);
        if ($idx >= $bins) $idx = $bins - 1;
        if ($idx < 0) $idx = 0;
        $counts[$idx]++;
    }
    
    return ['labels' => $labels, 'counts' => $counts, 'bin_width' => $bin_width, 'min' => $min_s];
}

// ============================================================
// RUN THE MODEL
// ============================================================

$n_samples = isset($_GET['samples']) ? intval($_GET['samples']) : 10000;
$seed = isset($_GET['seed']) ? intval($_GET['seed']) : 42;
if ($n_samples < 100) $n_samples = 100;
if ($n_samples > 100000) $n_samples = 100000;

$results = cholera_risk_assessment($n_samples, $seed);
$sensitivity = sensitivity_analysis(
    $results['P_final_samples'],
    $results['P2'],
    $results['P3'],
    $results['P4']
);
$histogram = build_histogram($results['P_final_samples'], 50);

// Compute summary stats for each step
$p2_mean = array_sum($results['P2']) / count($results['P2']);
$p3_mean = array_sum($results['P3']) / count($results['P3']);
$p4_mean = array_sum($results['P4']) / count($results['P4']);

// Subsample for scatter plot (too many points = slow)
$scatter_max = 500;
$scatter_step = max(1, intval($n_samples / $scatter_max));
$scatter_p2 = []; $scatter_pfinal = [];
for ($i = 0; $i < $n_samples; $i += $scatter_step) {
    $scatter_p2[] = $results['P2'][$i];
    $scatter_pfinal[] = $results['P_final_samples'][$i];
}

?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Cholera Risk Assessment - Monte Carlo Simulation for evaluating transboundary cholera infection risk from Bangladesh to India">
    <title>Cholera Risk Assessment | Monte Carlo Simulation</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
    <style>
        :root {
            --bg-primary: #f8fafc;
            --bg-secondary: #f1f5f9;
            --bg-card: #ffffff;
            --bg-card-hover: #f8fafc;
            --accent-blue: #2563eb;
            --accent-cyan: #0891b2;
            --accent-purple: #7c3aed;
            --accent-emerald: #059669;
            --accent-rose: #e11d48;
            --accent-amber: #d97706;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --border-color: #e2e8f0;
            --glow-blue: rgba(37,99,235,0.12);
            --glow-cyan: rgba(8,145,178,0.10);
            --radius: 16px;
            --radius-sm: 10px;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
            --shadow-md: 0 4px 14px rgba(0,0,0,0.06), 0 2px 6px rgba(0,0,0,0.04);
            --shadow-lg: 0 10px 30px rgba(0,0,0,0.08), 0 4px 10px rgba(0,0,0,0.04);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }

        /* Subtle gradient background */
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: 
                radial-gradient(ellipse at 20% 10%, rgba(37,99,235,0.04) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 90%, rgba(124,58,237,0.03) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(8,145,178,0.02) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }

        .app-container {
            position: relative;
            z-index: 1;
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px 24px 60px;
        }

        /* ── Header ── */
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 40px 20px;
            position: relative;
        }

        .header::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 120px;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
            border-radius: 3px;
        }

        .header-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 16px;
            background: rgba(37,99,235,0.08);
            border: 1px solid rgba(37,99,235,0.15);
            border-radius: 99px;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--accent-blue);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 16px;
        }

        .header-badge svg {
            width: 14px; height: 14px;
        }

        h1 {
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -1px;
            background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }

        .header-sub {
            font-size: 1.05rem;
            color: var(--text-secondary);
            max-width: 650px;
            margin: 0 auto;
        }

        /* ── Controls ── */
        .controls-panel {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            margin-bottom: 36px;
            flex-wrap: wrap;
        }

        .control-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .control-group label {
            font-size: 0.82rem;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .control-input {
            padding: 10px 16px;
            background: #fff;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            width: 130px;
            transition: border-color 0.3s, box-shadow 0.3s;
        }

        .control-input:focus {
            outline: none;
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px var(--glow-blue);
        }

        .btn-run {
            padding: 10px 28px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border: none;
            border-radius: var(--radius-sm);
            color: #fff;
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.3s, opacity 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            min-width: 170px;
            justify-content: center;
        }

        .btn-run:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 25px rgba(37,99,235,0.25);
        }

        .btn-run.loading {
            pointer-events: none;
            opacity: 0.85;
        }

        .btn-spinner {
            display: none;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: #fff;
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
        }

        .btn-run.loading .btn-spinner {
            display: inline-block;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* ── Stat Cards Grid ── */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 36px;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 22px 20px;
            position: relative;
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
            box-shadow: var(--shadow-sm);
        }

        .stat-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-md);
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
        }

        .stat-card:nth-child(1)::before { background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }
        .stat-card:nth-child(2)::before { background: linear-gradient(90deg, var(--accent-emerald), #34d399); }
        .stat-card:nth-child(3)::before { background: linear-gradient(90deg, var(--accent-purple), #a78bfa); }
        .stat-card:nth-child(4)::before { background: linear-gradient(90deg, var(--accent-amber), #fbbf24); }
        .stat-card:nth-child(5)::before { background: linear-gradient(90deg, var(--accent-rose), #fb7185); }

        .stat-label {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 8px;
        }

        .stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .stat-detail {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 4px;
        }

        /* ── Step Parameters ── */
        .parameters-section {
            margin-bottom: 36px;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section-title .dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--accent-cyan);
        }

        .param-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 14px;
        }

        .param-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            padding: 18px 20px;
            display: flex;
            align-items: flex-start;
            gap: 14px;
            transition: background 0.3s, box-shadow 0.3s;
            box-shadow: var(--shadow-sm);
        }

        .param-card:hover {
            background: var(--bg-card-hover);
            box-shadow: var(--shadow-md);
        }

        .param-step {
            min-width: 36px; height: 36px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 700;
            color: #fff;
            flex-shrink: 0;
        }

        .step-1 { background: linear-gradient(135deg, #3b82f6, #2563eb); }
        .step-2 { background: linear-gradient(135deg, #10b981, #059669); }
        .step-3 { background: linear-gradient(135deg, #f59e0b, #d97706); }
        .step-4 { background: linear-gradient(135deg, #ef4444, #dc2626); }
        .step-5 { background: linear-gradient(135deg, #8b5cf6, #7c3aed); }
        .step-6 { background: linear-gradient(135deg, #ec4899, #db2777); }

        .param-info h4 {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 3px;
            color: var(--text-primary);
        }

        .param-info p {
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-bottom: 4px;
        }

        .param-value-tag {
            display: inline-block;
            padding: 2px 10px;
            background: rgba(37,99,235,0.06);
            border: 1px solid rgba(37,99,235,0.12);
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--accent-blue);
        }

        /* ── Charts ── */
        .charts-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 36px;
        }

        @media (max-width: 900px) {
            .charts-section { grid-template-columns: 1fr; }
        }

        .chart-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 24px;
            position: relative;
            box-shadow: var(--shadow-sm);
        }

        .chart-card.full-width {
            grid-column: 1 / -1;
        }

        .chart-title {
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-primary);
        }

        .chart-title-icon {
            width: 28px; height: 28px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
        }

        .chart-title-icon.hist { background: rgba(37,99,235,0.1); color: var(--accent-blue); }
        .chart-title-icon.sens { background: rgba(225,29,72,0.1); color: var(--accent-rose); }
        .chart-title-icon.scatter { background: rgba(5,150,105,0.1); color: var(--accent-emerald); }
        .chart-title-icon.box { background: rgba(124,58,237,0.1); color: var(--accent-purple); }

        canvas {
            width: 100% !important;
            max-height: 400px;
        }

        /* ── Methodology ── */
        .methodology {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 30px;
            margin-bottom: 36px;
            box-shadow: var(--shadow-sm);
        }

        .methodology h3 {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 14px;
            color: var(--text-primary);
        }

        .formula-block {
            background: #f1f5f9;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            padding: 16px 20px;
            margin: 12px 0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            color: var(--accent-blue);
            overflow-x: auto;
        }

        .methodology p {
            font-size: 0.88rem;
            color: var(--text-secondary);
            line-height: 1.7;
            margin-bottom: 10px;
        }

        .methodology ul {
            padding-left: 20px;
            margin-bottom: 10px;
        }

        .methodology li {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }

        /* ── Footer ── */
        .footer {
            text-align: center;
            padding: 30px;
            color: var(--text-muted);
            font-size: 0.78rem;
            border-top: 1px solid var(--border-color);
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

        /* ── Animations ── */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .animate-in {
            animation: fadeInUp 0.6s ease-out both;
        }
        .delay-1 { animation-delay: 0.1s; }
        .delay-2 { animation-delay: 0.2s; }
        .delay-3 { animation-delay: 0.3s; }
        .delay-4 { animation-delay: 0.4s; }
        .delay-5 { animation-delay: 0.5s; }
    </style>
</head>
<body>
<div class="app-container">

    <!-- Header -->
    <header class="header animate-in">
        <div class="header-badge">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
            Monte Carlo Simulation
        </div>
        <h1>Cholera Risk Assessment</h1>
        <p class="header-sub">Quantitative risk analysis of transboundary cholera infection from Bangladesh to India using stochastic Monte Carlo simulation</p>
    </header>

    <!-- Controls -->
    <form method="GET" class="controls-panel animate-in delay-1" id="simForm">
        <div class="control-group">
            <label for="samples">Samples:</label>
            <input type="number" name="samples" id="samples" class="control-input" 
                   value="<?= htmlspecialchars($n_samples) ?>" min="100" max="100000" step="100">
        </div>
        <div class="control-group">
            <label for="seed">Seed:</label>
            <input type="number" name="seed" id="seed" class="control-input" 
                   value="<?= htmlspecialchars($seed) ?>" min="1" max="9999999">
        </div>
        <button type="submit" class="btn-run" id="btnRunSim">
            <span class="btn-spinner"></span>
            <span class="btn-text">▶ Run Simulation</span>
        </button>
    </form>

    <!-- Summary Statistics -->
    <div class="stats-grid animate-in delay-2">
        <div class="stat-card">
            <div class="stat-label">Expected Probability</div>
            <div class="stat-value"><?= sprintf("%.6e", $results['expected_prob']) ?></div>
            <div class="stat-detail">Mean of <?= number_format($n_samples) ?> samples</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">90% CI Lower</div>
            <div class="stat-value"><?= sprintf("%.2e", $results['ci'][0]) ?></div>
            <div class="stat-detail">5th percentile</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">90% CI Upper</div>
            <div class="stat-value"><?= sprintf("%.2e", $results['ci'][1]) ?></div>
            <div class="stat-detail">95th percentile</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Minimum</div>
            <div class="stat-value"><?= sprintf("%.2e", $results['min_val']) ?></div>
            <div class="stat-detail">Lowest simulated</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Maximum</div>
            <div class="stat-value"><?= sprintf("%.2e", $results['max_val']) ?></div>
            <div class="stat-detail">Highest simulated</div>
        </div>
    </div>

    <!-- Step Parameters -->
    <div class="parameters-section animate-in delay-3">
        <h2 class="section-title"><span class="dot"></span> Model Parameters</h2>
        <div class="param-grid">
            <div class="param-card">
                <div class="param-step step-1">P1</div>
                <div class="param-info">
                    <h4>Outbreak Probability (Bangladesh)</h4>
                    <p>Poisson process: 1 − e<sup>−λt</sup> | λ = 2779/(22×12)</p>
                    <span class="param-value-tag"><?= sprintf("%.6f", $results['P1']) ?></span>
                </div>
            </div>
            <div class="param-card">
                <div class="param-step step-2">P2</div>
                <div class="param-info">
                    <h4>Infection Rate (Beta Distribution)</h4>
                    <p>Beta(1605, 24618) – human infection probability</p>
                    <span class="param-value-tag">mean ≈ <?= sprintf("%.6f", $p2_mean) ?></span>
                </div>
            </div>
            <div class="param-card">
                <div class="param-step step-3">P3</div>
                <div class="param-info">
                    <h4>False Negative (Bangladesh)</h4>
                    <p>1 − Sensitivity(CholKit RDT): U(0.549, 0.906)</p>
                    <span class="param-value-tag">mean ≈ <?= sprintf("%.6f", $p3_mean) ?></span>
                </div>
            </div>
            <div class="param-card">
                <div class="param-step step-4">P4</div>
                <div class="param-info">
                    <h4>False Negative (India)</h4>
                    <p>1 − Sensitivity(CholKit RDT): U(0.884, 0.999)</p>
                    <span class="param-value-tag">mean ≈ <?= sprintf("%.6f", $p4_mean) ?></span>
                </div>
            </div>
            <div class="param-card">
                <div class="param-step step-5">P5</div>
                <div class="param-info">
                    <h4>Exposure Risk (Water & Sanitation)</h4>
                    <p>Unsafe water (11%) × unsafe sanitation (1%)</p>
                    <span class="param-value-tag"><?= sprintf("%.4f", $results['P5']) ?></span>
                </div>
            </div>
            <div class="param-card">
                <div class="param-step step-6">P6</div>
                <div class="param-info">
                    <h4>Mortality Probability</h4>
                    <p>Case fatality rate</p>
                    <span class="param-value-tag"><?= sprintf("%.2f", $results['P6']) ?></span>
                </div>
            </div>
        </div>
    </div>

    <!-- Charts -->
    <div class="charts-section animate-in delay-4">
        <!-- Histogram -->
        <div class="chart-card full-width">
            <div class="chart-title">
                <span class="chart-title-icon hist">📊</span>
                Distribution of Simulated Risk Probabilities
            </div>
            <canvas id="histogramChart"></canvas>
        </div>

        <!-- Sensitivity -->
        <div class="chart-card">
            <div class="chart-title">
                <span class="chart-title-icon sens">🎯</span>
                Sensitivity Analysis (Spearman ρ)
            </div>
            <canvas id="sensitivityChart"></canvas>
        </div>

        <!-- Scatter -->
        <div class="chart-card">
            <div class="chart-title">
                <span class="chart-title-icon scatter">🔬</span>
                P2 vs Final Risk (Scatter)
            </div>
            <canvas id="scatterChart"></canvas>
        </div>

        <!-- Box Plot approximation -->
        <div class="chart-card full-width">
            <div class="chart-title">
                <span class="chart-title-icon box">📦</span>
                Risk Distribution Summary
            </div>
            <canvas id="boxPlotChart"></canvas>
        </div>
    </div>

    <!-- Methodology -->
    <div class="methodology animate-in delay-5">
        <h3>📐 Methodology</h3>
        <p>This application performs a <strong>quantitative risk assessment</strong> of cholera transmission from Bangladesh to India using a stochastic Monte Carlo approach. The model multiplies six probability components:</p>
        <div class="formula-block">
            P_final = P1 × P2 × P3 × P4 × P5 × P6
        </div>
        <p>Where each component represents a step in the transmission chain, from outbreak occurrence to mortality outcome. The simulation generates <strong><?= number_format($n_samples) ?></strong> random samples by drawing from appropriate probability distributions:</p>
        <ul>
            <li><strong>P1:</strong> Deterministic from Poisson process (λ = <?= sprintf("%.2f", $results['lambda_rate']) ?>)</li>
            <li><strong>P2:</strong> Stochastic – Beta(1605, 24618) distribution</li>
            <li><strong>P3:</strong> Stochastic – 1 − Uniform(0.549, 0.906)</li>
            <li><strong>P4:</strong> Stochastic – 1 − Uniform(0.884, 0.999)</li>
            <li><strong>P5:</strong> Deterministic point estimate (0.11 × 0.01)</li>
            <li><strong>P6:</strong> Deterministic point estimate (0.03)</li>
        </ul>
        <p>Sensitivity analysis uses <strong>Spearman's rank correlation coefficient</strong> to quantify how each stochastic input influences the final risk output.</p>
    </div>

    <!-- Footer -->
    <footer class="footer">
        Cholera Risk Assessment &mdash; Monte Carlo Simulation Engine &mdash; PHP Implementation &copy; <?= date('Y') ?>
    </footer>

</div>

<script>
// ── Button Loading State ──
document.getElementById('simForm').addEventListener('submit', function() {
    const btn = document.getElementById('btnRunSim');
    btn.classList.add('loading');
    btn.querySelector('.btn-text').textContent = 'Processing...';
});

// ── Chart.js Configuration ──
Chart.defaults.color = '#475569';
Chart.defaults.borderColor = 'rgba(0,0,0,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";

// ── Data from PHP ──
const histLabels = <?= json_encode($histogram['labels']) ?>;
const histCounts = <?= json_encode($histogram['counts']) ?>;
const expectedProb = <?= $results['expected_prob'] ?>;
const minVal = <?= $results['min_val'] ?>;
const maxVal = <?= $results['max_val'] ?>;

const sensitivityLabels = <?= json_encode(array_keys($sensitivity)) ?>;
const sensitivityValues = <?= json_encode(array_values($sensitivity)) ?>;

const scatterP2 = <?= json_encode($scatter_p2) ?>;
const scatterPF = <?= json_encode($scatter_pfinal) ?>;

// Percentile data for box plot
const ci5 = <?= $results['ci'][0] ?>;
const ci95 = <?= $results['ci'][1] ?>;
const p25 = <?= percentile($results['P_final_samples'], 25) ?>;
const p50 = <?= percentile($results['P_final_samples'], 50) ?>;
const p75 = <?= percentile($results['P_final_samples'], 75) ?>;

// ── 1. Histogram ──
const histCtx = document.getElementById('histogramChart').getContext('2d');

// Find bin index closest to expected, min, max for annotation
function findClosestBin(val) {
    let minDist = Infinity, idx = 0;
    histLabels.forEach((l, i) => {
        const d = Math.abs(parseFloat(l) - val);
        if (d < minDist) { minDist = d; idx = i; }
    });
    return idx;
}

// Find bin index closest to a value
const meanBinIdx = findClosestBin(expectedProb);

new Chart(histCtx, {
    type: 'bar',
    data: {
        labels: histLabels,
        datasets: [{
            label: 'Frequency',
            data: histCounts,
            backgroundColor: 'rgba(37,99,235,0.45)',
            borderColor: 'rgba(37,99,235,0.7)',
            borderWidth: 1,
            borderRadius: 2,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: { display: false },
            tooltip: {
                callbacks: {
                    title: (items) => `Risk ≈ ${items[0].label}`,
                    label: (item) => `Count: ${item.raw}`
                }
            },
            annotation: {
                annotations: {
                    meanLine: {
                        type: 'line',
                        xMin: meanBinIdx,
                        xMax: meanBinIdx,
                        borderColor: 'rgba(220, 38, 38, 0.9)',
                        borderWidth: 2.5,
                        borderDash: [6, 3],
                        label: {
                            display: true,
                            content: `Mean = ${expectedProb.toExponential(2)}`,
                            position: 'start',
                            backgroundColor: 'rgba(220, 38, 38, 0.85)',
                            color: '#fff',
                            font: { size: 11, weight: '600', family: "'JetBrains Mono', monospace" },
                            padding: { x: 8, y: 4 },
                            borderRadius: 4,
                        }
                    }
                }
            }
        },
        scales: {
            x: {
                title: { display: true, text: 'Probability of Cholera Infection (India from Bangladesh)', font: { size: 12 }, color: '#334155' },
                ticks: { 
                    maxTicksLimit: 8,
                    font: { size: 10 },
                    color: '#64748b'
                },
                grid: { display: false }
            },
            y: {
                title: { display: true, text: 'Frequency', font: { size: 12 }, color: '#334155' },
                beginAtZero: true,
                grid: { color: 'rgba(0,0,0,0.05)' },
                ticks: { color: '#64748b' }
            }
        }
    }
});

// ── 2. Sensitivity Analysis (Horizontal Bar) ──
const sensCtx = document.getElementById('sensitivityChart').getContext('2d');
new Chart(sensCtx, {
    type: 'bar',
    data: {
        labels: sensitivityLabels,
        datasets: [{
            label: 'Spearman ρ',
            data: sensitivityValues,
            backgroundColor: sensitivityValues.map(v => v > 0 
                ? 'rgba(37,99,235,0.5)' 
                : 'rgba(225,29,72,0.5)'),
            borderColor: sensitivityValues.map(v => v > 0 
                ? 'rgba(37,99,235,0.9)' 
                : 'rgba(225,29,72,0.9)'),
            borderWidth: 1,
            borderRadius: 4,
        }]
    },
    options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: { display: false },
            tooltip: {
                callbacks: {
                    label: (item) => `ρ = ${item.raw.toFixed(4)}`
                }
            }
        },
        scales: {
            x: {
                title: { display: true, text: 'Spearman Rank Correlation', font: { size: 11 } },
                grid: { color: 'rgba(0,0,0,0.05)' },
                suggestedMin: -1,
                suggestedMax: 1,
            },
            y: {
                grid: { display: false }
            }
        }
    }
});

// ── 3. Scatter Plot ──
const scatCtx = document.getElementById('scatterChart').getContext('2d');
const scatterData = scatterP2.map((v, i) => ({ x: v, y: scatterPF[i] }));
new Chart(scatCtx, {
    type: 'scatter',
    data: {
        datasets: [{
            label: 'P2 vs P_final',
            data: scatterData,
            backgroundColor: 'rgba(5,150,105,0.3)',
            borderColor: 'rgba(5,150,105,0.7)',
            pointRadius: 2.5,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: { display: false },
            tooltip: {
                callbacks: {
                    label: (item) => `P2=${item.parsed.x.toExponential(3)}, Risk=${item.parsed.y.toExponential(3)}`
                }
            }
        },
        scales: {
            x: {
                title: { display: true, text: 'P2 (Infection Rate)', font: { size: 11 } },
                grid: { color: 'rgba(0,0,0,0.05)' }
            },
            y: {
                title: { display: true, text: 'Final Risk', font: { size: 11 } },
                grid: { color: 'rgba(0,0,0,0.05)' }
            }
        }
    }
});

// ── 4. Box Plot Approximation ──
const boxCtx = document.getElementById('boxPlotChart').getContext('2d');
new Chart(boxCtx, {
    type: 'bar',
    data: {
        labels: ['Risk Distribution'],
        datasets: [
            {
                label: 'Min to 5th %ile',
                data: [ci5 - minVal],
                backgroundColor: 'rgba(100,116,139,0.2)',
                borderWidth: 0,
                barPercentage: 0.4,
            },
            {
                label: '5th to 25th %ile',
                data: [p25 - ci5],
                backgroundColor: 'rgba(59,130,246,0.3)',
                borderWidth: 0,
                barPercentage: 0.4,
            },
            {
                label: '25th to Median',
                data: [p50 - p25],
                backgroundColor: 'rgba(59,130,246,0.6)',
                borderWidth: 0,
                barPercentage: 0.4,
            },
            {
                label: 'Median to 75th %ile',
                data: [p75 - p50],
                backgroundColor: 'rgba(139,92,246,0.6)',
                borderWidth: 0,
                barPercentage: 0.4,
            },
            {
                label: '75th to 95th %ile',
                data: [ci95 - p75],
                backgroundColor: 'rgba(139,92,246,0.3)',
                borderWidth: 0,
                barPercentage: 0.4,
            },
            {
                label: '95th %ile to Max',
                data: [maxVal - ci95],
                backgroundColor: 'rgba(100,116,139,0.2)',
                borderWidth: 0,
                barPercentage: 0.4,
            },
        ]
    },
    options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: { 
                display: true,
                position: 'bottom',
                labels: { 
                    font: { size: 10 },
                    boxWidth: 12,
                    padding: 15
                }
            },
            tooltip: {
                callbacks: {
                    label: function(ctx) {
                        const labels = ['Min', '5th', '25th', 'Median', '75th', '95th', 'Max'];
                        const values = [minVal, ci5, p25, p50, p75, ci95, maxVal];
                        const start = values[ctx.datasetIndex];
                        const end = values[ctx.datasetIndex + 1];
                        return `${labels[ctx.datasetIndex]} (${start.toExponential(2)}) → ${labels[ctx.datasetIndex+1]} (${end.toExponential(2)})`;
                    }
                }
            }
        },
        scales: {
            x: {
                stacked: true,
                title: { display: true, text: 'Risk Probability', font: { size: 11 } },
                grid: { color: 'rgba(0,0,0,0.05)' }
            },
            y: {
                stacked: true,
                grid: { display: false }
            }
        }
    }
});
</script>
</body>
</html>
