style = """
QMainWindow {
    background-color: #fafafa;
}

QGroupBox {
    background-color: #ffffff;
    color: #333333;
    border: 1px solid #dddddd;
    border-radius: 10px;
    font-weight: bold;
    margin-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 4px 12px;
}

QPushButton {
    background-color: #e6e6e6;
    color: #333333;
    border-radius: 8px;
    padding: 6px 14px;
}
QPushButton:hover {
    background-color: #d0d0d0;
}

QLabel, QLineEdit, QComboBox {
    color: #333333;
}
QLineEdit, QComboBox {
    background-color: #ffffff;
    border: 1px solid #cccccc;
    border-radius: 6px;
    padding: 4px;
}

QSlider::groove:horizontal {
    background: #e0e0e0;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #007aff;
    width: 16px;
    border-radius: 8px;
}

QProgressBar {
    background-color: #e6e6e6;
    color: #333333;
    border-radius: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #007aff;
    border-radius: 8px;
}
"""
