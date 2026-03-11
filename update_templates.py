import re

def update_file(filepath, pattern, replacement):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

# Update base.html
base_style = """    <!-- Use Inter font for premium feel -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            /* Premium Color Palette */
            --primary: #667eea;
            --primary-hover: #5a6fd8;
            --secondary: #764ba2;
            --background: #fdfdfd;
            --surface: rgba(255, 255, 255, 0.85);
            --surface-solid: #ffffff;
            --text-main: #2d3748;
            --text-muted: #718096;
            --border: #e2e8f0;
            --border-focus: #cbd5e0;
            
            --success: #38a169;
            --danger: #e53e3e;
            --warning: #dd6b20;

            --radius-sm: 6px;
            --radius-md: 12px;
            --radius-lg: 16px;
            
            --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            --shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
            --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            
            --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #f4f7f6;
            background-image: radial-gradient(circle at 100% 0%, #f1f2fc 0%, #f4f7f6 50%, #fdfdfd 100%);
            color: var(--text-main);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px 15px;
        }

        .header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 24px 0;
            margin-bottom: 24px;
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-md);
            position: relative;
            overflow: hidden;
        }

        .header h1 {
            text-align: center;
            font-size: 32px;
            font-weight: 700;
            letter-spacing: -0.5px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .controls {
            background: var(--surface);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 24px;
            border-radius: var(--radius-lg);
            border: 1px solid rgba(255,255,255,0.4);
            box-shadow: var(--shadow-md);
            margin-bottom: 30px;
        }

        .control-group {
            margin-bottom: 20px;
        }

        .control-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            font-size: 14px;
            color: var(--text-main);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .control-row {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
        }

        input[type="password"], input[type="text"], input[type="url"], select {
            padding: 10px 16px;
            border: 2px solid var(--border);
            border-radius: var(--radius-md);
            font-size: 15px;
            font-family: inherit;
            background: var(--surface-solid);
            transition: var(--transition);
            color: var(--text-main);
            box-shadow: 0 1px 2px rgba(0,0,0,0.02) inset;
        }

        input[type="password"]:focus, input[type="text"]:focus, input[type="url"]:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
            outline: none;
        }

        .checkbox-group {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }

        .checkbox-item {
            position: relative;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: var(--surface-solid);
            padding: 8px 16px;
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
            cursor: pointer;
            transition: var(--transition);
            font-weight: 500;
            font-size: 14px;
        }
        
        .checkbox-item:hover {
            border-color: var(--border-focus);
            background: #f8fafc;
        }
        
        .checkbox-item input[type="checkbox"] {
            width: 18px;
            height: 18px;
            accent-color: var(--primary);
            cursor: pointer;
        }

        .filter-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
        }

        .slider-container {
            margin-top: 15px;
            padding: 0 5px;
        }

        .slider-container input[type="range"] {
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: #e2e8f0;
            outline: none;
            -webkit-appearance: none;
            transition: background 0.2s;
        }

        .slider-container input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
            border: 4px solid white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            transition: transform 0.1s ease;
        }
        
        .slider-container input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.15);
        }

        .slider-container input[type="range"]::-moz-range-thumb {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
            border: 4px solid white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            transition: transform 0.1s ease;
        }
        
        .slider-container input[type="range"]::-moz-range-thumb:hover {
            transform: scale(1.15);
        }

        .slider-labels {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            margin-top: 10px;
            font-size: 13px;
            color: var(--text-muted);
            font-weight: 500;
        }

        .slider-labels span:nth-child(1) { text-align: left; }
        .slider-labels span:nth-child(2) { text-align: center; }
        .slider-labels span:nth-child(3) { text-align: right; }

        #scoreThresholdValue {
            color: var(--primary);
            font-weight: 700;
            font-size: 15px;
            background: rgba(102, 126, 234, 0.1);
            padding: 4px 10px;
            border-radius: 12px;
        }

        .btn {
            background: var(--primary);
            color: white;
            padding: 10px 24px;
            border: none;
            border-radius: var(--radius-md);
            cursor: pointer;
            font-size: 15px;
            font-weight: 600;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition);
            box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
        }

        .btn:hover {
            background: var(--primary-hover);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
        }

        .success { color: var(--success); font-weight: 600; }
        .error { color: var(--danger); font-weight: 600; }

        .table-container {
            background: var(--surface-solid);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-lg);
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .table-scroll {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }

        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 14px;
        }

        th {
            background: #f8fafc;
            padding: 16px 12px;
            text-align: left;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
            border-bottom: 2px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 10;
            white-space: nowrap;
        }

        td {
            padding: 14px 12px;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
            transition: background 0.15s ease;
        }

        tbody tr:not(.details-row):hover {
            background-color: #f1f5f9;
        }
        
        tbody tr.active-details {
            background-color: #ebf4ff !important;
            border-left: 3px solid var(--primary) !important;
        }

        .rating-select {
            width: 100%;
            min-width: 80px;
            padding: 8px;
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            font-size: 14px;
            background: var(--surface-solid);
            cursor: pointer;
            box-shadow: var(--shadow-sm);
        }

        .listing-link {
            color: var(--primary);
            text-decoration: none;
            font-weight: 600;
        }

        .listing-link:hover {
            text-decoration: underline;
            color: var(--secondary);
        }

        .score { font-weight: 700; color: var(--success); }
        .price { font-weight: 600; color: var(--text-main); font-size: 15px;}

        @media (max-width: 768px) {
            .container { padding: 10px; }
            .header h1 { font-size: 24px; }
            .controls { padding: 16px; margin-bottom: 20px;}
            table { font-size: 13px; }
            th, td { padding: 10px 6px; }
            .control-row { flex-direction: column; align-items: stretch; }
            .filter-row { grid-template-columns: 1fr; gap: 16px; }
            .property-details > div { grid-template-columns: 1fr !important; gap: 12px !important; }
            .location-cell { word-wrap: break-word; overflow-wrap: break-word; max-width: 120px; line-height: 1.4; }
        }

        @media (max-width: 600px) { .hide-mobile { display: none; } }
        @media (max-width: 480px) {
            .hide-very-small { display: none; }
            .show-on-very-small { display: block !important; }
            th, td { padding: 8px 4px !important; font-size: 12px !important; }
            .location-cell { max-width: 90px; }
            .price { font-size: 13px !important; }
        }
        @media (max-width: 768px) { .hide-mobile-kitchen { display: none; } }

        .rating-column { width: 65px !important; min-width: 65px !important; max-width: 65px !important; text-align: center; }
        .rating-select { width: 55px !important; min-width: 55px !important; max-width: 55px !important; padding: 6px !important; text-align: center; border-radius: 6px !important; font-size: 14px !important;}
        .info-column { width: 45px !important; min-width: 45px !important; max-width: 45px !important; text-align: center; }

        @media (max-width: 480px) {
            .rating-column { width: 50px !important; min-width: 50px !important; max-width: 50px !important; padding: 2px !important; }
            .rating-select { width: 44px !important; min-width: 44px !important; max-width: 44px !important; padding: 4px !important; font-size: 12px !important; }
            .info-column { width: 35px !important; min-width: 35px !important; max-width: 35px !important; padding: 2px !important; }
        }

        .no-data { text-align: center; padding: 40px 20px; color: var(--text-muted); font-size: 16px; font-weight: 500;}

        .scroll-to-top {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 50px;
            height: 50px;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            font-size: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            opacity: 0;
            visibility: hidden;
            transform: translateY(20px);
            transition: var(--transition);
            z-index: 1000;
        }

        .scroll-to-top.visible { opacity: 1; visibility: visible; transform: translateY(0); }
        .scroll-to-top:hover { transform: translateY(-4px); background: var(--secondary); box-shadow: 0 8px 16px rgba(118, 75, 162, 0.4); }

        .details-row {
            transition: all 0.3s ease;
        }
        
        .details-row td {
            background: #fafafa;
            border-bottom: 2px solid var(--border);
        }

        .property-details {
            max-width: 100%;
            overflow-x: auto;
        }

        @media (max-width: 768px) {
            .scroll-to-top { bottom: 20px; right: 20px; width: 44px; height: 44px; font-size: 18px; }
        }
    </style>"""

update_file('c:/Users/shawn/House/templates/base.html', r'<style>.*?</style>', base_style)

# Update index.html
index_style = """<style>
    /* Navigation links */
    .nav-links-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        flex-wrap: wrap;
    }

    .nav-link {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 20px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        text-decoration: none;
        border-radius: var(--radius-md);
        font-weight: 600;
        transition: var(--transition);
        box-shadow: var(--shadow-sm);
    }

    .nav-link:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        color: white;
    }

    .nav-separator {
        font-size: 18px;
        color: var(--text-muted);
        font-weight: bold;
    }

    .ai-ranking-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px;
    }

    .toggle-container {
        display: flex;
        align-items: center;
        gap: 16px;
        background: var(--surface-solid);
        padding: 12px 24px;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
    }

    .toggle-switch {
        position: relative;
        display: inline-block;
        width: 60px;
        height: 34px;
    }

    .toggle-switch input {
        opacity: 0;
        width: 0;
        height: 0;
    }

    .toggle-slider {
        position: absolute;
        cursor: pointer;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: #cbd5e0;
        transition: .4s;
        border-radius: 34px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }

    .toggle-slider:before {
        position: absolute;
        content: "";
        height: 26px;
        width: 26px;
        left: 4px;
        bottom: 4px;
        background-color: white;
        transition: .4s;
        border-radius: 50%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    input:checked + .toggle-slider {
        background: linear-gradient(135deg, var(--success) 0%, #20c997 100%);
    }

    input:checked + .toggle-slider:before {
        transform: translateX(26px);
    }

    input:disabled + .toggle-slider {
        background-color: #e2e8f0;
        cursor: not-allowed;
    }

    input:disabled:checked + .toggle-slider {
        background: #9ae6b4;
    }

    .toggle-label {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .toggle-label strong {
        font-size: 15px;
        color: var(--text-main);
    }

    .toggle-description {
        font-size: 12px;
        color: var(--text-muted);
        font-style: auto;
        max-width: 250px;
    }

    .ai-ranking-status {
        font-size: 13px;
        font-weight: 600;
        text-align: center;
        min-height: 18px;
        color: var(--primary);
    }

    .add-property-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
    }

    .add-property-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 12px 24px;
        background: linear-gradient(135deg, var(--success) 0%, #20c997 100%);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        font-weight: 600;
        font-size: 15px;
        cursor: pointer;
        transition: var(--transition);
        box-shadow: var(--shadow-sm);
    }

    .add-property-btn:hover:not(:disabled) {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }

    .add-property-btn:disabled {
        background: var(--border-focus);
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
    }

    .add-property-hint {
        font-size: 12px;
        color: var(--text-muted);
        text-align: center;
    }

    /* Modal styles */
    .modal {
        display: none;
        position: fixed;
        z-index: 1000;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(4px);
    }

    .modal-content {
        background-color: var(--surface-solid);
        margin: 10% auto;
        padding: 24px;
        border: none;
        border-radius: var(--radius-lg);
        width: 90%;
        max-width: 500px;
        box-shadow: var(--shadow-lg);
        transform: translateY(0);
        animation: modalSlideIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    @keyframes modalSlideIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--border);
    }

    .modal-header h3 {
        margin: 0;
        color: var(--text-main);
        font-size: 20px;
        font-weight: 700;
    }

    .close {
        color: var(--text-muted);
        font-size: 28px;
        font-weight: bold;
        cursor: pointer;
        line-height: 1;
        transition: color 0.2s;
    }

    .close:hover { color: var(--danger); }

    .modal-body { margin-bottom: 24px; }
    .modal-body label {
        display: block;
        margin-bottom: 8px;
        font-weight: 600;
        color: var(--text-main);
    }

    .modal-body input[type="url"] {
        width: 100%;
        padding: 12px;
        border: 2px solid var(--border);
        border-radius: var(--radius-md);
        font-size: 15px;
        box-sizing: border-box;
        transition: var(--transition);
    }

    .modal-body input[type="url"]:focus {
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
    }

    .modal-footer {
        display: flex;
        justify-content: flex-end;
        gap: 12px;
    }

    .modal-btn {
        padding: 10px 20px;
        border: none;
        border-radius: var(--radius-sm);
        font-size: 15px;
        font-weight: 600;
        cursor: pointer;
        transition: var(--transition);
    }

    .modal-btn-primary { background: var(--primary); color: white; }
    .modal-btn-primary:hover { background: var(--primary-hover); }
    .modal-btn-secondary { background: #e2e8f0; color: var(--text-main); }
    .modal-btn-secondary:hover { background: #cbd5e0; }

    @media (max-width: 768px) {
        .nav-links-container { flex-direction: column; gap: 10px; }
        .nav-link { width: 100%; max-width: 250px; justify-content: center; }
        .nav-separator { display: none; }
        .toggle-container { flex-direction: column; gap: 12px; text-align: center; }
        .toggle-description { max-width: 300px; }
        .add-property-btn { width: 100%; justify-content: center; }
        .modal-content { margin: 5% auto; width: 95%; }
    }
</style>"""

update_file('c:/Users/shawn/House/templates/index.html', r'<style>.*?</style>', index_style)

# Inject showDetails JS changes to add/remove active row state
with open('c:/Users/shawn/House/templates/index.html', 'r', encoding='utf-8') as f:
    js_content = f.read()

# Make the expandable row have an active tr class
js_content = js_content.replace(
    "detailsRow.style.display = 'table-row';", 
    "detailsRow.style.display = 'table-row';\n                const mainRow = document.querySelector(`select[onchange*=\"${zpid}\"]`).closest('tr');\n                if(mainRow) mainRow.classList.add('active-details');"
)
js_content = js_content.replace(
    "detailsRow.style.display = 'none';", 
    "detailsRow.style.display = 'none';\n                const mainRow = document.querySelector(`select[onchange*=\"${zpid}\"]`)?.closest('tr');\n                if(mainRow) mainRow.classList.remove('active-details');"
)

with open('c:/Users/shawn/House/templates/index.html', 'w', encoding='utf-8') as f:
    f.write(js_content)

print("Updated templates")
