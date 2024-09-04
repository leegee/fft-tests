# src/SpectrogramDelegate.py

from PySide6.QtWidgets import QStyledItemDelegate, QLabel, QStyleOptionViewItem
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtCore import QSize, Qt, QRect, QEvent

class SpectrogramDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        if index.column() == 2:  # Assuming column 2 is for spectrogram
            pixmap = QPixmap(index.data())  # Fetch the image path
            if not pixmap.isNull():
                # Paint the pixmap in the cell
                rect = self.getAdjustedRect(option)
                painter.drawPixmap(rect, pixmap)
        else:
            super().paint(painter, option, index)
    
    def sizeHint(self, option, index):
        if index.column() == 2:
            return QSize(100, 100)  # Set size hint for the image
        else:
            return super().sizeHint(option, index)
    
    def getAdjustedRect(self, option):
        rect = option.rect
        # Adjust the rect to ensure proper alignment
        rect.adjust(5, 5, -5, -5)
        return rect
    
    def editorEvent(self, event, model, option, index):
        if event.type() == QEvent.MouseButtonRelease and index.column() == 2:
            # Handle click event
            rect = self.getAdjustedRect(option)
            if rect.contains(event.position().toPoint()):  # Use event.position().toPoint()
                # Emit a signal or perform an action
                print(f"Clicked on spectrogram in row {index.row()}")
        return super().editorEvent(event, model, option, index)
