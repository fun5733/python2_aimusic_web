#-*-coding:utf-8
from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = {
    path('', views.index, name="index"),
    path('2', views.test, name="po1"),
    path('3', views.down, name="down"),
}