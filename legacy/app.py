import os
import logging

logger = logging.getLogger(__file__)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QFileDialog, QLineEdit, QLabel
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import QSize

from indexer import index_images
from matcher import semantic_search

class MainWindow(QMainWindow):
    def __init__(self, win_res=(1200, 800)):
        logger.info("Hello!!!")
        super(MainWindow, self).__init__()
        self.setFixedSize(QSize(*win_res))

        work_dir = os.path.join(os.path.dirname(__file__), '.simgs_cache')
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        self.work_dir = work_dir

        # TODO: multiple appropriately used functions later
        button = QPushButton("Index a Folder")
        button.clicked.connect(self.button_click)

        button2 = QPushButton("Search Using Keyword")
        button2.clicked.connect(self.search)

        self.kw_input = QLineEdit()
        self.img_view = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(button)
        layout.addWidget(self.kw_input)
        layout.addWidget(button2)
        layout.addWidget(self.img_view)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

    def search(self):
        logger.info("searing for `%s`" % self.kw_input.text())
        res = semantic_search(self.work_dir, self.kw_input.text(), topK=1)
        if res:
            img_path = res[0]
        qimg = QPixmap(img_path)
        self.img_view.setPixmap(qimg)
    
    def button_click(self):
        file_diag = QFileDialog(self)
        file_diag.setFileMode(QFileDialog.FileMode.Directory)
        file_diag.exec()

        selected_folders = file_diag.selectedFiles()
        for ff in selected_folders:
            index_images(ff, self.work_dir)



def get_res(screen_width, screen_height):
    # TODO: complete this; consider tall screens too
    return (1200, 800)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(process)d - %(filename)s:%(lineno)d - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
        )
    logger.info("Starting...")
    # TODO: rewrite such that initial model download happens after UI/splashscreen loads
    app = QApplication([])

    main_win = MainWindow()
    main_win.show()

    app.exec()
