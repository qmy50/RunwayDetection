#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>
#include <QWidget>
#include <QPushButton>
#include <QProcess>
#include <QDebug>

namespace Ui {
class Dialog;
}

class Dialog : public QDialog
{
    Q_OBJECT

public:
    explicit Dialog(QWidget *parent = nullptr);
    enum Mode{
        ImageMode,
        VideoMode,
        NoMode
    };
    Mode getSelectMode()const{
        return selectedMode;
    }
    ~Dialog();
signals:
    void modelSelected(Mode mode);
private slots:
    void onImageModeClicked();
    void onVideoModeCLicked();

private:
    Ui::Dialog *ui;
    QPushButton *imageButton;
    QPushButton *videoButton;
    Mode selectedMode;
};

#endif // DIALOG_H
