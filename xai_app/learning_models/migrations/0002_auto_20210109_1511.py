# Generated by Django 3.1.5 on 2021-01-09 15:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('learning_models', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='learningmodel',
            name='title',
            field=models.CharField(max_length=128, unique=True, verbose_name='название'),
        ),
    ]
