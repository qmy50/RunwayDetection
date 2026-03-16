#include "widget.h"
#include "dialog.h"
#include "imagewidget.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Widget w;
    ImageWidget iw;
    Dialog dlg;
    if(dlg.exec() != QDialog::Accepted){
        return 0;
    }

    Dialog::Mode mode = dlg.getSelectMode();
    if (mode == Dialog::VideoMode){
        w.show();
        return a.exec();
    }else if(mode == Dialog::ImageMode){
       iw.show();
       return a.exec();
    }
    else{
        return 0;
    }

}
