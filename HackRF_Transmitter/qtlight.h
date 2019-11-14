#ifndef LIGHTWIDGET_H
#define LIGHTWIDGET_H

#include <QWidget>
#include <QPainter>

class QtLight : public QWidget
{
    Q_OBJECT
    Q_PROPERTY(bool on READ isOn WRITE setOn)
public:
    QtLight(QWidget *parent = 0)
    : QWidget(parent)
    , m_color(Qt::red)
    , m_on(false)
    {

    }

    bool isOn() const
    {
        return m_on;
    }

    void setOn(bool on)
    {
        if (on == m_on)
            return;
        m_on = on;
        update();
    }

public slots:
    void turnOff() { setOn(false); }
    void turnOn() { setOn(true); }

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        if(!m_on)
        {
            painter.setBrush(Qt::red);
        }
        else
        {
            painter.setBrush(Qt::green);
        }
        painter.drawEllipse(0, 0, width(), height());
    }

private:
    QColor m_color;
    bool m_on;
};


#endif // LIGHTWIDGET_H
