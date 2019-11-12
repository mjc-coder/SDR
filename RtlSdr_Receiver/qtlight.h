#ifndef QTLIGHT_H
#define QTLIGHT_H

#include <QWidget>
#include <QPainter>

class QtLight : public QWidget
{
    Q_OBJECT

public:
    enum QTLight_State
    {
        BAD = 0,
        WARN = 1,
        GOOD = 2,
        NONE = 3
    };
public:
    QtLight(QWidget *parent = 0)
    : QWidget(parent)
    , m_color(Qt::yellow)
    , m_on(NONE)
    {

    }

    void setOn(QTLight_State on)
    {
        m_on = on;
        update();
    }

    QTLight_State getState() const
    {
        return m_on;
    }

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        if(m_on == BAD)
        {
            painter.setBrush(Qt::red);
        }
        else if(m_on == WARN)
        {
            painter.setBrush(Qt::yellow);
        }
        else if(m_on == GOOD)
        {
            painter.setBrush(Qt::green);
        }
        else
        {
            painter.setBrush(Qt::gray);
        }
        painter.drawEllipse(0, 0, width(), height());
    }

private:
    QColor m_color;
    QTLight_State m_on;
};


#endif // LIGHTWIDGET_H

