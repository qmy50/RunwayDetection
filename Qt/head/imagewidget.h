#ifndef IMAGEWIDGET_H
#define IMAGEWIDGET_H

#include <QWidget>
#include <QFileInfo>
#include <QVBoxLayout>
#include <QFile>
#include <QVBoxLayout>
#include <QMessageBox>
#include <QString>
#include <QFileDialog>
#include <QPixmap>
#include "widget.h"

namespace Ui {
class ImageWidget;
}

class ImageWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ImageWidget(QWidget *parent = nullptr);
    ~ImageWidget();

private slots:
    void on_selectButton_clicked();

    void on_solveButton_clicked();

    void on_reSelectButton_clicked();

private:
    Ui::ImageWidget *ui;
    QString image_path;
};

#endif // IMAGEWIDGET_H
