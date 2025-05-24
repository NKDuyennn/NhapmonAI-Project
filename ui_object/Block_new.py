from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QFont, QPalette
from PyQt5.QtCore import Qt

class Block(QLabel):
    def __init__(self, number, size, pixmap=None):
        super(Block, self).__init__()
        self.number = number
        self.pixmap = pixmap
        self.setFixedSize(size, size)
        
        # Create a label for the image or number
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(size, size)
        
        if number == 0:
            # Empty tile: transparent or blank
            if pixmap:
                self.label.setPixmap(pixmap.scaled(size, size, Qt.KeepAspectRatio))
            else:
                self.setStyleSheet("background-color: transparent;")
        else:
            # Display image piece with number overlay
            if pixmap:
                self.label.setPixmap(pixmap.scaled(size, size, Qt.KeepAspectRatio))
                # Add number overlay in bottom-left corner
                number_label = QLabel(str(number), self)
                number_label.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
                # Set smaller size for number_label to fit in bottom-left corner
                number_label.setFixedSize(size // 2, size // 2)
                # Position the label in the bottom-left corner
                number_label.move(5, size - (size // 2) - 5)  # Small padding
                font = QFont()
                font.setPointSize(size // 8)  # Smaller font size
                font.setBold(True)
                number_label.setFont(font)
                # Set bold orange color with semi-transparent background
                number_label.setStyleSheet("color: #000000; background-color: rgba(0, 0, 0, 0);")
            else:
                # Fallback to number-only display
                self.label.setText(str(number))
                font = QFont()
                font.setPointSize(size // 4)
                font.setBold(True)
                self.label.setFont(font)
                self.setStyleSheet("background-color: lightblue; border: 1px solid black;")
