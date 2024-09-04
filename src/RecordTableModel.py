# src/RecordTableModel.py

from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex

class RecordTableModel(QAbstractTableModel):
    def __init__(self, records=None, parent=None):
        super().__init__(parent)
        self.records = records if records is not None else []

    def rowCount(self, parent=QModelIndex()):
        return len(self.records)

    def columnCount(self, parent=QModelIndex()):
        return 3  # ID, Filename, Spectrogram

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        column = index.column()

        if role == Qt.DisplayRole:
            record = self.records[row]
            if column == 0:
                return record[0]  # ID
            elif column == 1:
                return record[1]  # Filename
            elif column == 2:
                plot_path = record[1].replace('.wav', '.png')
                return plot_path  # Return the image path for the delegate to use

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                headers = ["ID", "Filename", "Spectrogram"]
                return headers[section]
        return None

    def setRecords(self, records):
        self.beginResetModel()
        self.records = records
        self.endResetModel()

