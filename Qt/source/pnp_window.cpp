#include "pnp_window.h"
#include "ui_pnp_window.h"

PNP_Window::PNP_Window(QString path, QWidget *parent) :
    QWidget(parent),
    viedoPath(path),
    ui(new Ui::PNP_Window)
{
    ui->setupUi(this);
//    runPythonScripts();
}

PNP_Window::~PNP_Window()
{
    delete ui;
}

void PNP_Window::runPythonScripts()
{
    QProcess *p = new QProcess(this);
    QString pythonExe = "D:\\anaconda\\envs\\gpu42\\python.exe";
    QString scriptPath = "D:\\vscode\\opencv_learning\\Runway_Project\\PNP_algorithm\\PNP_1.py";
    QStringList args;
    args << "-u" << scriptPath << viedoPath;

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("CONDA_DEFAULT_ENV", "gpu42"); // 显式指定conda环境
    p->setProcessEnvironment(env);

    connect(p,&QProcess::readyReadStandardOutput,[=](){
        qDebug() << "进行Python程序调用";
    });

    connect(p,
            static_cast<void(QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished),
            [=](int exitCode, QProcess::ExitStatus exitStatus) {
                qDebug() << "退出码：" << exitCode;
                qDebug() << "退出状态：" << exitStatus;
                p->deleteLater();
                if (exitStatus == QProcess::NormalExit){
                    qDebug() << "程序正常退出";
                    displayResult();
                }
            });

    connect(p, &QProcess::readyReadStandardOutput, [=](){
        QByteArray output = p->readAllStandardOutput();
        qDebug() << "[Python输出]:" << QString(output).trimmed(); // 必须读取output
    });

    p->start(pythonExe,args);

    connect(p, &QProcess::readyReadStandardError, [=](){
        QByteArray error = p->readAllStandardError();
        QString errLog = QString::fromLocal8Bit(error).trimmed()
                         .replace("\\r\\n", "\n")
                         .replace("\\\\", "\\")
                         .replace("\"", "");
        qDebug() << "[Python错误]:" << qPrintable(errLog);  // 打印cv2导入错误
    });
}

void PNP_Window::on_pushButton_clicked()
{
    qDebug() << "即将开始Python调用" ;
    runPythonScripts();
}

void PNP_Window::displayResult()
{
    if (m_player == nullptr) {
        m_player = new QMediaPlayer(this); // 父对象为this，自动管理内存
        m_player->setNotifyInterval(30);   // 可选：设置通知间隔
    }

    QFileInfo fileInfo(return_add_vie);
    if (!fileInfo.exists()) {
        qDebug() << "错误：视频文件不存在！路径：" << return_add_vie;
        return;
    }

    QVideoWidget *videoWin = new QVideoWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(ui->axes_widget);
    layout->setContentsMargins(0, 0, 0, 0); // 去掉布局边距，视频填满widget
    layout->addWidget(videoWin);
    m_player->setVideoOutput(videoWin);
    m_player->setMedia(QMediaContent(QUrl::fromLocalFile(return_add_vie)));
    qDebug() << "即将开始播放视频";
    connect(m_player, static_cast<void (QMediaPlayer::*)(QMediaPlayer::MediaStatus)>(&QMediaPlayer::mediaStatusChanged),
            this, [=](QMediaPlayer::MediaStatus status) {
        // 视频加载完成后自动播放
        if (status == QMediaPlayer::LoadedMedia) {
            m_player->play();
        }
    });
    connect(m_player, static_cast<void (QMediaPlayer::*)(QMediaPlayer::Error)>(&QMediaPlayer::error),
            this, [=](QMediaPlayer::Error error) {
        if (error != QMediaPlayer::NoError) {
            qDebug() << "视频播放错误：" << m_player->errorString();
        }
    });

    connect(ui->pushButton_2,&QPushButton::clicked,this,&PNP_Window::on_restartButton_3_clicked);

    if (m_imgLabel == nullptr) {
        m_imgLabel = new QLabel(this);
        m_imgLabel->setAlignment(Qt::AlignCenter); // 图片居中显示
        m_imgLabel->setScaledContents(true); // 图片自适应QLabel大小（关键）

        QVBoxLayout *layout = new QVBoxLayout(ui->plot_widget_2);
        layout->setContentsMargins(0, 0, 0, 0); // 去掉边距，填满widget
        layout->addWidget(m_imgLabel);
    }

    // 加载图片并显示
    QPixmap pixmap(return_add_img);
    if (pixmap.isNull()) {
        qDebug() << "错误：图片加载失败（格式不支持/文件损坏）！";
        return;
    }

    pixmap = pixmap.scaled(ui->axes_widget->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);

    // 设置图片到QLabel
    m_imgLabel->setPixmap(pixmap);
    qDebug() << "图片显示成功：";
}

void PNP_Window::on_restartButton_3_clicked()
{
     m_player->setPosition(0);
    if(m_player->state() != QMediaPlayer::PlayingState){
        m_player->play();
        qDebug() << "视频开始播放";
    }
}
