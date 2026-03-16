#include "imagewidget.h"
#include "ui_imagewidget.h"
#include "QDebug"
#include "dialog.h"

ImageWidget::ImageWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ImageWidget)
{
    ui->setupUi(this);
}

ImageWidget::~ImageWidget()
{
    delete ui;
}

void ImageWidget::on_selectButton_clicked()
{
    image_path = QFileDialog::getOpenFileName(
                this,                          // 父窗口，对话框居中显示
                tr("选择图片文件"),             // 对话框标题
                QDir::homePath(),              // 默认打开的路径（桌面：QDir::desktopPath()）
                tr("视频文件 (*.jpg *.png);;所有文件 (*.*)") // 过滤支持的视频格式
            );
    QPixmap pixmap(image_path);

    ui->label->setPixmap(pixmap.scaled(ui->label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    ui->label->setAlignment(Qt::AlignCenter);
    if (image_path.isEmpty()){
        QMessageBox::warning(this,"警告","请选择一个文件");
        return;
    }
}

void ImageWidget::on_solveButton_clicked()
{
    qDebug()<<"调用图片姿态解算程序";
}

void ImageWidget::on_reSelectButton_clicked()
{
    Dialog dlg(this);
   int result = dlg.exec();  // 模态阻塞

   if (result == QDialog::Accepted) {
       Dialog::Mode mode = dlg.getSelectMode();
       if (mode == Dialog::ImageMode) {
           QMessageBox::information(this, "提示", "当前已在图片模式，继续。");
       } else if (mode == Dialog::VideoMode) {
           qDebug() << "即将进入视频模式" ;
           Widget* widget = new Widget();
           widget->show();
           close();  // 关闭 Widget
       }
   }
}
