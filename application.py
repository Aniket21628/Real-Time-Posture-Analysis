import sys

import qdarktheme
from PyQt5.QtWidgets import QApplication

from app_controllers.controller import Controller
from app_models.model import Model
from app_views.view import View

model_name = "small640.pt"


class App:
    def __init__(self):
        super().__init__()
        self.model = Model(model_name)
        self.view = View(self.model)
        self.controller = Controller(self.model, self.view)
        print("All modules loaded")


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Error: Exactly one argument is expected")
        sys.exit(1)
    elif len(sys.argv) == 1:
        print("Info: Loading default inference model: {}".format(model_name))
    else:
        model_name = sys.argv[1]
    qdarktheme.enable_hi_dpi()
    app = QApplication([])
    qdarktheme.setup_theme('dark')
    # update global stylesheet with modern teal theme
    current_stylesheet = app.styleSheet()
    updated_stylesheet = current_stylesheet + """
        QMainWindow {background-color: #1a1a2e;}
        QSlider {background-color: #16213e; border-radius: 5px;}
        QGroupBox {
            background-color: #16213e;
            border: 2px solid #0f4c75;
            border-radius: 8px;
            margin-top: 10px;
            font-size: 13px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #3bd6c6;
        }
        QLabel {font-weight: bold; color: #e8e8e8;}
        QPushButton{font-weight: bold;}
        QCheckBox {font-weight: bold; color: #e8e8e8;}
        QRadioButton {font-weight: bold; color: #e8e8e8;}
        QComboBox {
            font-weight: bold;
            background-color: #16213e;
            border: 2px solid #0f4c75;
            border-radius: 5px;
            padding: 5px;
            color: #e8e8e8;
        }
        QComboBox:hover {
            border: 2px solid #3bd6c6;
        }
        QToolTip {
            background-color: #16213e;
            color: #3bd6c6;
            font-size: 12px;
            border: 2px solid #0f4c75;
            border-radius: 6px;
            padding: 8px;
        }
    """
    app.setStyleSheet(updated_stylesheet)
    window = App()
    window.view.show()
    sys.exit(app.exec())
