#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QMediaPlayer>
#include <QMediaContent>
#include <QDebug>
#include <QUrl>
#include <QFileInfo>
#include <QVBoxLayout>
#include <QFile>
#include <QMessageBox>
#include <pnp_window.h>
#include "imagewidget.h"

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
    QString getTime(qint64 cur_time);
    QString viedoPath_f;

//private slots:
//    void on_volumehorizontalSlider_2_actionTriggered(int action);

private slots:
    void on_pausepushButton_3_clicked();
    void on_playpushButton_2_clicked();
    void onDurationChanged();
    void onPositionChanged(qint64 position);
    void on_selectpushButton_clicked();

    void on_reSelectButton_clicked();

private:
    Ui::Widget *ui;
    QMediaPlayer* m_player;
    QString total_time_string;
    QString viedo_path;
};
#endif // WIDGET_H
