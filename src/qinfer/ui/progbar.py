# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'progbar.ui'
#
# Created: Tue Feb 19 17:36:00 2013
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from __future__ import absolute_import
from PySide import QtCore, QtGui

class Ui_ProgBarDialog(object):
    def setupUi(self, ProgBarDialog):
        ProgBarDialog.setObjectName("ProgBarDialog")
        ProgBarDialog.resize(451, 184)
        self.verticalLayout = QtGui.QVBoxLayout(ProgBarDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lbl_task_title = QtGui.QLabel(ProgBarDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_task_title.sizePolicy().hasHeightForWidth())
        self.lbl_task_title.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setWeight(75)
        font.setBold(True)
        self.lbl_task_title.setFont(font)
        self.lbl_task_title.setObjectName("lbl_task_title")
        self.horizontalLayout.addWidget(self.lbl_task_title)
        spacerItem = QtGui.QSpacerItem(8, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.line = QtGui.QFrame(ProgBarDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line.sizePolicy().hasHeightForWidth())
        self.line.setSizePolicy(sizePolicy)
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.lbl_task_status = QtGui.QLabel(ProgBarDialog)
        self.lbl_task_status.setObjectName("lbl_task_status")
        self.verticalLayout.addWidget(self.lbl_task_status)
        self.prog_bar = QtGui.QProgressBar(ProgBarDialog)
        self.prog_bar.setProperty("value", 0)
        self.prog_bar.setFormat("")
        self.prog_bar.setObjectName("prog_bar")
        self.verticalLayout.addWidget(self.prog_bar)
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lbl_eta = QtGui.QLabel(ProgBarDialog)
        self.lbl_eta.setEnabled(True)
        self.lbl_eta.setObjectName("lbl_eta")
        self.horizontalLayout_2.addWidget(self.lbl_eta)
        self.buttonBox = QtGui.QDialogButtonBox(ProgBarDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel)
        self.buttonBox.setObjectName("buttonBox")
        self.horizontalLayout_2.addWidget(self.buttonBox)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(ProgBarDialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), ProgBarDialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), ProgBarDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(ProgBarDialog)

    def retranslateUi(self, ProgBarDialog):
        ProgBarDialog.setWindowTitle(QtGui.QApplication.translate("ProgBarDialog", "Task Title", None, QtGui.QApplication.UnicodeUTF8))
        self.lbl_task_title.setText(QtGui.QApplication.translate("ProgBarDialog", "Task Title", None, QtGui.QApplication.UnicodeUTF8))
        self.lbl_task_status.setText(QtGui.QApplication.translate("ProgBarDialog", "Task Status...", None, QtGui.QApplication.UnicodeUTF8))
        self.lbl_eta.setText(QtGui.QApplication.translate("ProgBarDialog", "Estimated time remaining:", None, QtGui.QApplication.UnicodeUTF8))

