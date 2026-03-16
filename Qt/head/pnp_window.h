#ifndef PNP_WINDOW_H
#define PNP_WINDOW_H

#include <QWidget>
#include <QProcess>
#include <QDebug>

#include <QMediaPlayer>
#include <QVideoWidget>
#include <QVBoxLayout>
#include <QUrl>
#include <QFileInfo>
#include <QLabel>
#include <QPixmap>

namespace Ui {
class PNP_Window;
}

class PNP_Window : public QWidget
{
    Q_OBJECT
public:
    explicit PNP_Window(QString path, QWidget *parent = nullptr);
    ~PNP_Window();
    QString viedoPath;
    QString return_add_img = "D:\\qt\\projects\\others\\trajectory.png";
    QString return_add_vie = "D:\\qt\\projects\\others\\save_viedo.avi";

private:
    Ui::PNP_Window *ui;
    QMediaPlayer *m_player = nullptr;
     QLabel *m_imgLabel = nullptr;
    void displayResult();

private slots:
    void runPythonScripts();
    void on_pushButton_clicked();
    void on_restartButton_3_clicked();
};

#endif // PNP_WINDOW_H
