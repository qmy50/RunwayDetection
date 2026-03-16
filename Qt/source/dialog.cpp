#include "dialog.h"
#include "ui_dialog.h"

Dialog::Dialog(QWidget *parent) :
    QDialog(parent),ui(new Ui::Dialog), selectedMode(NoMode)
{
    setWindowTitle("请选择模式");
    setModal(true);  // 模态对话框
    ui->setupUi(this);
    connect(ui->imagePushButton, &QPushButton::clicked, this, &Dialog::onImageModeClicked);
    connect(ui->videoPushButton, &QPushButton::clicked, this, &Dialog::onVideoModeCLicked);
}

Dialog::~Dialog()
{
    delete ui;
}

void Dialog::onImageModeClicked()
{
    selectedMode = ImageMode;
   qDebug() << "当前选择图片模式";
    accept();  // 关闭对话框，返回 Accepted
}

void Dialog::onVideoModeCLicked()
{
    selectedMode = VideoMode;
    qDebug() << "当前选择视频模式";
    accept();
}
