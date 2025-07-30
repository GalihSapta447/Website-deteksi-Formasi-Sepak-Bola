import streamlit as st
import os
import numpy as np
import json
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import logging
import sqlite3
import json

# Konfigurasi awal
HISTORY_FILE = 'history.json'
VISUAL_PATH = 'Static/images'
CLASS_NAMES = ['4-3-3', '4-4-2', '3-5-2']

# Load model
url = 'https://drive.google.com/file/d/19fDI1OvYIPkvjhdnRsCQ9Eu8vWdI3rHn'
output = 'Model/CNN_formasi.h5'
gdown.download(url, output, quiet=False)

logging.basicConfig(
    filename='Logs/app.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#custom CSS
def load_css(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        logging.error("File CSS tidak ditemukan")
        st.error("❌ Gagal memuat file CSS")

load_css("CSS/styles.css")

# Inisialisasi database
def init_db():
    try:
        conn = sqlite3.connect('Data/history.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    prediction TEXT,
                    confidence REAL,
                    timestamp TEXT
                    )''')
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"SQLite error: {e}")
        st.error("❌ Gagal menginisialisasi database")
    finally:
        conn.close()
init_db()

# Fungsi prediksi
def predict_formation(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    max_index = np.argmax(prediction)
    confidence = float(round(prediction[max_index] * 100, 2))
    return CLASS_NAMES[max_index], confidence

# Fungsi untuk menggambar formasi
def draw_formation(formation_name):
    fig, ax = plt.subplots(figsize=(8, 10), facecolor="#588157")
    ax.set_facecolor("#588157")

    formation_positions = {
        '4-3-3': {
            'GK': [(50, 5)],
            'CB': [(40, 30), (60, 30)],
            'RB': [(75, 35)],
            'LB': [(25, 35)],
            'CMF': [(35, 60), (50, 55), (65, 60)],
            'LWF': [(30, 100)],
            'RWF': [(70, 100)],
            'CF': [(50, 110)]
        },
        '4-4-2': {
            'GK': [(50, 5)],
            'CB': [(40, 30), (60, 30)],
            'RB': [(75, 35)],
            'LB': [(25, 35)],
            'CMF': [(40, 60), (60, 60)],
            'LMF': [(25, 65)],
            'RMF': [(75, 65)],
            'CF': [(40, 100), (60, 100)]
        },
        '3-5-2': {
            'GK': [(50, 5)],
            'CB': [(35, 30), (50, 30), (65, 30)],
            'CMF': [(35, 60), (50, 55), (65, 60)], 
            'LMF': [(20, 70)],
            'RMF': [(80, 70)],
            'CF': [(40, 100), (60, 100)]
        }
    }

    positions = formation_positions.get(formation_name, {})

    for role, coords_list in positions.items():
        color = 'yellow' if role == 'GK' else 'red' if role == 'CB' else 'blue' if role == 'CMF' else 'blue' if role == 'RMF' else 'blue' if role == 'LMF' else 'red' if role == 'RB' else 'red' if role == 'LB' else 'orange'
        for x, y in coords_list:
            ax.add_patch(plt.Circle((x, y), 3, color=color, zorder=5))
            ax.text(x, y + 5, role, color='white', fontsize=8, ha='center', va='bottom')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 130)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    plt.title(f"Visualisasi Formasi {formation_name}", color='white', fontsize=28)
    return fig

# Simpan riwayat
def save_history(filename, prediction, confidence):
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    except:
        history = []

    entry = {
        'id': len(history),
        'filename': filename,
        'prediction': str(prediction),
        'confidence': float(confidence),
        'timestamp': datetime.now().isoformat()
    }

    history.append(entry)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

# Tampilkan riwayat
def load_history():
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

# Hapus item riwayat berdasarkan index
def delete_history_item(index):
    try:
        history = load_history()
        if 0 <= index < len(history):
            history.pop(index)
            with open(HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
            return True
    except:
        pass
    return False

# Hapus semua riwayat
def clear_all_history():
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)
        return True
    except:
        return False

# Format tanggal
def format_datetime(timestamp):
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%d %b %Y, %H:%M")
    except:
        return timestamp

# Fungsi untuk mendapatkan statistik
def get_statistics():
    history = load_history()
    total_detections = len(history)
    
    if total_detections == 0:
        return {
            'total': 0,
            'formations': {},
            'avg_confidence': 0,
            'today': 0
        }
    
    formations = {}
    confidences = []
    today = datetime.now().date()
    today_count = 0
    
    for entry in history:
        formation = entry['prediction']
        formations[formation] = formations.get(formation, 0) + 1
        confidences.append(entry['confidence'])
        
        try:
            entry_date = datetime.fromisoformat(entry['timestamp']).date()
            if entry_date == today:
                today_count += 1
        except:
            pass
    
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        'total': total_detections,
        'formations': formations,
        'avg_confidence': avg_confidence,
        'today': today_count
    }

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Formasi Sepak Bola", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 2rem; info-card;">
        <h2 style="color: white; margin: 0;">⚽ Menu Navigasi</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Menu options
    menu_options = {
        "🏠 Beranda": "home",
        "📸 Deteksi Formasi": "detect",
        "📊 Dashboard": "dashboard",
        "📝 Riwayat": "history",
    }
    
    # Navigation menu
    selected_page = st.selectbox(
        "Pilih Halaman:",
        list(menu_options.keys()),
        key="navigation"
    )
    
    page_key = menu_options[selected_page]
    
    st.markdown("---")
    
    # Quick stats in sidebar
    stats = get_statistics()
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">{stats['total']}</div>
        <div class="stats-label">Total Deteksi</div>
    </div>
    """, unsafe_allow_html=True)
    
    if stats['total'] > 0:
        most_common = max(stats['formations'], key=stats['formations'].get) if stats['formations'] else "Belum ada"
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{most_common}</div>
            <div class="stats-label">Formasi Terpopuler</div>
        </div>
        """, unsafe_allow_html=True)

# Main content based on selected page
if page_key == "home":
    # Beranda
    st.markdown("""
    <div class="main-header fade-in">
        <h1 class="main-title">⚽ Ruang Formasi</h1>
        <p class="main-subtitle">Deteksi Formasi Sepak Bola Menggunakan AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h2 class="page-title">🏠Tentang Website</h2>
        <p class="page-description">Website ini bertujuan untuk mengenali pola formasi sepak bola dengan teknologi Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #667eea; text-align: center;">⚡ Cepat & Mudah</h3>
            <p style="text-align: center;">Unggah gambar dan dapatkan hasil deteksi formasi dalam hitungan detik</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #667eea; text-align: center;">📊 Visualisasi</h3>
            <p style="text-align: center;">Lihat visualisasi formasi yang terdeteksi dalam bentuk gambar pola formasi</p>
        </div>
        """, unsafe_allow_html=True)

elif page_key == "detect":
    # Halaman Deteksi
    st.markdown("""
    <div class="info-card">
        <h2 class="page-title">📸 Deteksi Formasi</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 style="color: #333;"> Unggah Gambar Pertandingan</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: #666;">Pilih gambar yang menunjukkan pola formasi untuk dideteksi secara otomatis.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag & drop atau klik untuk memilih gambar", 
        type=["jpg", "jpeg", "png"],
        help="Format yang didukung: JPG, JPEG, PNG"
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="Gambar yang Diunggah", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("🔎 Deteksi Formasi", key="predict_btn"):
            with st.spinner("Sedang mengdeteksi gambar..."):
                prediction, confidence = predict_formation(img)
                
                st.markdown(f"""
                <div class="result-card fade-in">
                    <h2 style="margin-bottom: 1rem;">🎯 Hasil Deteksi</h2>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; flex-wrap: wrap;">
                        <div>
                            <h3 style="margin: 0;">Formasi Terdeteksi:</h3>
                            <h1 style="margin: 0.5rem 0; font-size: 3rem;">{prediction}</h1>
                        </div>
                        <div>
                            <h3 style="margin: 0;">Tingkat Kepercayaan:</h3>
                            <h1 style="margin: 0.5rem 0; font-size: 3rem;">{confidence:.1f}%</h1>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Tingkat Kepercayaan")
                progress_col1, progress_col2 = st.columns([3, 1])
                with progress_col1:
                    st.progress(int(confidence))
                with progress_col2:
                    st.markdown(f"**{confidence:.1f}%**")
                
                save_history(uploaded_file.name, prediction, confidence)
                
                st.markdown('<h3 style="color: #333;">📊 Visualisasi Formasi</h3>', unsafe_allow_html=True)
                
                st.markdown('<h4 style="color: #333; text-align: center;">⚽ Pola Formasi:</h4>', unsafe_allow_html=True)
                fig_formation = draw_formation(prediction)
                
                col_viz1, col_viz2, col_viz3 = st.columns([1, 2, 1])
                with col_viz2:
                    st.pyplot(fig_formation)
                
                visual_file = os.path.join(VISUAL_PATH, f"{prediction}.png")
                if os.path.exists(visual_file):
                    st.markdown('<h4 style="color: #333; text-align: center;">🖼️ Gambar Formasi:</h4>', unsafe_allow_html=True)
                    col_static1, col_static2, col_static3 = st.columns([1, 2, 1])
                    with col_static2:
                        st.image(visual_file, caption=f"Pola Formasi {prediction}", use_container_width=True)
                else:
                    st.warning("⚠️ Gambar visualisasi formasi tidak ditemukan.")
                
                st.success("✅ Deteksi selesai! Hasil telah disimpan dalam riwayat.")
                
    else:
        st.markdown("</div>", unsafe_allow_html=True)

elif page_key == "dashboard":
    # Dashboard
    st.markdown("""
    <div class="info-card">
        <h2 class="page-title">📊 Dashboard</h2>
        <p class="page-description">Statistik dan analisis deteksi formasi</p>
    </div>
    """, unsafe_allow_html=True)
    
    stats = get_statistics()
    
    # Stats Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{stats['total']}</div>
            <div class="stats-label">Total Deteksi</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        most_common = max(stats['formations'], key=stats['formations'].get) if stats['formations'] else "Belum ada"
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{most_common}</div>
            <div class="stats-label">Formasi Terpopuler</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{stats['avg_confidence']:.1f}%</div>
            <div class="stats-label">Rata-rata Akurasi</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{stats['today']}</div>
            <div class="stats-label">Deteksi Hari Ini</div>
        </div>
        """, unsafe_allow_html=True)
    
    if stats['total'] > 0:
        # Formation Distribution
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #667eea;">📈 Ringkasan Formasi</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for formation, count in stats['formations'].items():
            percentage = (count / stats['total']) * 100
            st.markdown(f"""
            <div style="margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-weight: 600;">{formation}</span>
                    <span>{count} deteksi ({percentage:.1f}%)</span>
                </div>
                <div style="background: #f0f0f0; border-radius: 10px; height: 10px;">
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height: 100%; width: {percentage}%; border-radius: 10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #667eea; text-align: center;">📊 Belum Ada Data</h3>
            <p style="text-align: center; color: #666;">Lakukan deteksi formasi untuk melihat statistik</p>
        </div>
        """, unsafe_allow_html=True)

elif page_key == "history":
    # Riwayat
    st.markdown("""
    <div class="info-card">
        <h2 class="page-title">📝 Riwayat Deteksi</h2>
    </div>
    """, unsafe_allow_html=True)
    
    history = load_history()

    if history:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<p style="color: #666; info-card;">Berikut adalah riwayat deteksi yang telah dilakukan:</p>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="clear-all-btn">', unsafe_allow_html=True)
            if st.button("🗑️ Hapus Semua", key="clear_all", help="Hapus semua riwayat"):
                if clear_all_history():
                    st.success("✅ Semua riwayat berhasil dihapus!")
                    st.rerun()
                else:
                    st.error("❌ Gagal menghapus riwayat!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        recent_history = list(reversed(history[-20:]))  # Show more items in history page
        
        for i, entry in enumerate(recent_history):
            confidence_display = f"{entry['confidence']:.1f}%"
            formatted_time = format_datetime(entry['timestamp'])
            original_index = len(history) - 1 - i
            
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"""
                <div class="history-item">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <strong>📄 {entry['filename']}</strong><br>
                            <small>🕐 {formatted_time}</small>
                        </div>
                        <div style="text-align: right;">
                            <span class="formation-badge">{entry['prediction']}</span>
                            <span class="confidence-badge">{confidence_display}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="delete-btn">', unsafe_allow_html=True)
                if st.button("🗑️", key=f"delete_{original_index}", help="Hapus item ini"):
                    if delete_history_item(original_index):
                        st.success("✅ Item berhasil dihapus!")
                        st.rerun()
                    else:
                        st.error("❌ Gagal menghapus item!")
                st.markdown('</div>', unsafe_allow_html=True)
        
        if len(history) > 20:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; color: #666; font-style: italic;">
                Menampilkan 20 riwayat terbaru dari total {len(history)} riwayat
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.markdown("""
        <div class="info-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #667eea;">📋 Belum ada riwayat deteksi</h3>
            <p style="color: #666;">Lakukan deteksi formasi terlebih dahulu untuk melihat riwayat</p>
        </div>
        """, unsafe_allow_html=True)

