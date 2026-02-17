import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel

class MinhaJanela(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Exemplo PyQt5 com Dark Mode")
        self.setGeometry(100, 100, 400, 300)

        # Widget central
        widget_central = QWidget()
        self.setCentralWidget(widget_central)

        # Layout vertical
        self.layout = QVBoxLayout()

        # Label
        self.label = QLabel("Clique no botão!")
        self.layout.addWidget(self.label)

        # Botão para ação
        self.botao = QPushButton("Clique Aqui")
        self.botao.clicked.connect(self.botao_clicado)
        self.layout.addWidget(self.botao)

        # Botão para alternar tema
        self.botao_tema = QPushButton("Ativar Dark Mode")
        self.botao_tema.clicked.connect(self.alternar_tema)
        self.layout.addWidget(self.botao_tema)

        # Define o layout no widget central
        widget_central.setLayout(self.layout)

        # Estado inicial do tema
        self.modo_escuro = False
        self.aplicar_tema()

    def botao_clicado(self):
        self.label.setText("Botão clicado!")

    def alternar_tema(self):
        # Alterna o estado do tema
        self.modo_escuro = not self.modo_escuro
        self.botao_tema.setText("Ativar Light Mode" if self.modo_escuro else "Ativar Dark Mode")
        self.aplicar_tema()

    def aplicar_tema(self):
        if self.modo_escuro:
            # Estilo para o dark mode
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2b2b2b;
                }
                QLabel {
                    color: #ffffff;
                    font-size: 14px;
                }
                QPushButton {
                    background-color: #555555;
                    color: #ffffff;
                    border: 1px solid #888888;
                    padding: 5px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #666666;
                }
            """)
        else:
            # Estilo para o light mode
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f0f0f0;
                }
                QLabel {
                    color: #000000;
                    font-size: 14px;
                }
                QPushButton {
                    background-color: #e0e0e0;
                    color: #000000;
                    border: 1px solid #888888;
                    padding: 5px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
            """)

# Inicializa a aplicação
app = QApplication(sys.argv)
janela = MinhaJanela()
janela.show()
sys.exit(app.exec_())