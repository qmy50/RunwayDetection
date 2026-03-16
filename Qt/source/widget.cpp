#include "widget.h"
#include "ui_widget.h"
#include "dialog.h"
#include <QtMultimedia/QtMultimedia>
#include <QtMultimediaWidgets>

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
    ,m_player(new QMediaPlayer(this))
{
    ui->setupUi(this);
    QVideoWidget *videoWin = new QVideoWidget(this);
//    videoWin->resize(400,300);

    QVBoxLayout *layout = new QVBoxLayout(ui->widget);
    layout->setContentsMargins(0, 0, 0, 0); // 去掉布局边距，视频填满 widget
    layout->addWidget(videoWin); // 将 QVideoWidget 添加到布局

    m_player->setVideoOutput(videoWin);
    m_player->setNotifyInterval(30);

//    connect(ui->selectpushButton,&QPushButton::clicked,this,&Widget::on_selectpushButton_clicked);
//    m_player->setMedia(QMediaContent(QUrl::fromLocalFile("C:\\Users\\qmy\\Downloads\\35173696672-1-192.mp4")));
    //进度条初始化
    ui->timehorizontalSlider->setRange(0,0);
    ui->timehorizontalSlider->setTracking(true);

    connect(m_player,&QMediaPlayer::durationChanged,this,[=](qint64 duration){
        ui->timehorizontalSlider->setRange(0,duration);
        qDebug() << "视频总时长（毫秒）"<<duration;
        total_time_string = getTime(duration);
    });

    connect(m_player, SIGNAL(durationChanged(qint64)), this, SLOT(onDurationChanged()));
    connect(m_player, SIGNAL(positionChanged(qint64)), this, SLOT(onPositionChanged(qint64)));
    connect(m_player, &QMediaPlayer::positionChanged, this, [=](qint64 position) {
        // 避免拖动进度条时的循环触发
        if (!ui->timehorizontalSlider->isSliderDown()) {
            ui->timehorizontalSlider->setValue(position);
        }
    });

    connect(ui->timehorizontalSlider, &QSlider::sliderReleased, this, [=]() {
        m_player->setPosition(ui->timehorizontalSlider->value());
        qDebug() << "跳转到进度：" << ui->timehorizontalSlider->value() << "毫秒";
    });

    connect(ui->volumehorizontalSlider_2,&QSlider::sliderReleased,this, [=](){
        m_player->setVolume(ui->volumehorizontalSlider_2->value());
        qDebug() << QString("将音量设置为%1\%").arg(ui->volumehorizontalSlider_2->value());
    });

    //暂停按钮
    connect(ui->pausepushButton_3,&QPushButton::clicked,this,&Widget::on_pausepushButton_3_clicked);

    //播放按钮
    connect(ui->playpushButton_2,&QPushButton::clicked,this,&Widget::on_playpushButton_2_clicked);

    // 监听加载状态
    connect(m_player,
            static_cast<void (QMediaPlayer::*)(QMediaPlayer::MediaStatus)>(&QMediaPlayer::mediaStatusChanged),
            this,
            [=](QMediaPlayer::MediaStatus status) {
                if (status == QMediaPlayer::LoadedMedia) {
                    qDebug() <<"设置音量";
                    m_player->setVolume(50); // 设置音量
                    m_player->play();
                }
            });

    connect(m_player,
            static_cast<void (QMediaPlayer::*)(QMediaPlayer::Error)>(&QMediaPlayer::error),
            this,
            [=](QMediaPlayer::Error error) {
                if (error != QMediaPlayer::NoError) {
                    qDebug() << "播放错误：" << m_player->errorString();
                }
            });
    connect(ui->PNPpushButton_4,&QPushButton::clicked,this,[this](){
        qDebug() << "调用pnp解算" ;
        PNP_Window* pnp_window = new PNP_Window(viedoPath_f);
        pnp_window->show();
    });
}

Widget::~Widget()
{
    delete ui;
}

void Widget::on_pausepushButton_3_clicked()
{
    if(m_player->state() == QMediaPlayer::PlayingState){
        m_player->pause();
        qDebug() << "视频暂停";
    }
}

void Widget::on_playpushButton_2_clicked()
{
    if(m_player->state() != QMediaPlayer::PlayingState){
        m_player->play();
        qDebug() << "视频开始播放";
    }
}

void Widget::onDurationChanged()
{
    QString currentTime = getTime(0);
    ui->actimelabel_2->setText(QString("%1 / %2").arg(currentTime).arg(total_time_string));
}

void Widget::onPositionChanged(qint64 position)
{
    if (!ui->timehorizontalSlider->isSliderDown()) {
        ui->timehorizontalSlider->setSliderPosition(position);
    }
    // 转换当前进度为时分秒
    QString currentTime = getTime(position);
    // 转换总时长为时分秒
    ui->actimelabel_2->setText(QString("%1 / %2").arg(currentTime).arg(total_time_string));
}

QString Widget::getTime(qint64 time)
{
    int minute = time / (60 * 1000);
    int second = time % (60 * 1000) / 1000;
//    qDebug() << second / 1000;
    QString result = {};
    if(minute  < 10){
        result += "0";
    }
    result += QString::number(minute);
    result += ':';
    if(second < 10){
        result += '0';
    }
    result += QString::number(second);
    return result;
}

void Widget::on_selectpushButton_clicked()
{
    viedo_path = QFileDialog::getOpenFileName(
                this,                          // 父窗口，对话框居中显示
                tr("选择视频文件"),             // 对话框标题
                QDir::homePath(),              // 默认打开的路径（桌面：QDir::desktopPath()）
                tr("视频文件 (*.avi *.mp4 *.mov *.wmv);;所有文件 (*.*)") // 过滤支持的视频格式
            );
    if (viedo_path.isEmpty()){
        QMessageBox::warning(this,"警告","请选择一个文件");
        return;
    }
    m_player->stop();
    qDebug() << "视频路径为："<<viedo_path;
    m_player->setMedia(QMediaContent(QUrl::fromLocalFile(viedo_path)));
    viedoPath_f = viedo_path;
}

void Widget::on_reSelectButton_clicked()
{
    Dialog dlg(this);
   int result = dlg.exec();  // 模态阻塞

   if (result == QDialog::Accepted) {
       Dialog::Mode mode = dlg.getSelectMode();
       if (mode == Dialog::VideoMode) {
           QMessageBox::information(this, "提示", "当前已在视频模式，继续");
       } else if (mode == Dialog::ImageMode) {
           qDebug() << "即将进入图片模式" ;
           ImageWidget* image_widget = new ImageWidget();
           image_widget->show();
           close();  // 关闭 Widget
       }
   }
}
