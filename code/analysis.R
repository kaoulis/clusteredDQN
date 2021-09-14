rm(list = ls())
setwd("D:/Kaoulis/Desktop/PycharmProjects/DeepQnetworks/results")

library(dplyr)
library(ggplot2)

cl1 <- read.csv("cl1.csv")
rec4 <- read.csv("rec4.csv")
rec8 <- read.csv("rec8.csv")
norm4 <- read.csv("norm4.csv")
norm8 <- read.csv("norm8.csv")

cl1$clusters <- 1
rec4$clusters <- 4
rec8$clusters <- 8
norm4$clusters <- 4
norm8$clusters <- 8

cl1$clusters <- as.factor(cl1$cluster)
rec4$clusters <- as.factor(rec4$clusters)
rec8$clusters <- as.factor(rec8$clusters)
norm4$clusters <- as.factor(norm4$clusters)
norm8$clusters <- as.factor(norm8$clusters)

cl1$sampling <- 'random'
rec4$sampling <- 'rec'
rec8$sampling <- 'rec'
norm4$sampling <- "norm"
norm8$sampling <- "norm"

cl1$sampling <- as.factor(cl1$sampling)
rec4$sampling <- as.factor(rec4$sampling)
rec8$sampling <- as.factor(rec8$sampling)
norm4$sampling <- as.factor(norm4$sampling)
norm8$sampling <- as.factor(norm8$sampling)

df1 <- bind_rows(cl1, rec4, rec8, norm4, norm8)
df1$clusters <- factor(df1$clusters, levels=unique(df1$clusters))
df1$sampling <- factor(df1$sampling, levels=c("random", "rec", "norm"))
df2 <-df1[!(df1$e == 0),]
ggplot(df2) +
  geom_boxplot(aes(x = clusters, y  = s, fill = sampling)) +
  labs(x='Clusters', y='Score') +
  ylim(0, 1000)

##########################################################################
############################ Barplot #####################################

library(tidyverse)
df2 %>% 
  group_by(clusters, sampling) %>% 
  summarise(AverageMpg = round(mean(s))) %>% 
  ggplot(aes(clusters, AverageMpg, label=AverageMpg, fill=sampling)) +
  labs(x='Clusters', y='Average Score') +
  geom_col() +
  geom_text(nudge_y = 0.5) + 
  facet_grid(~sampling, scales = "free_x", space = "free_x")

##########################################################################
############################ Cardiogram ####################################

ggplot() +
  geom_line(aes(x=e, y=avg, color='cl1'), data=cl1) +
  geom_line(aes(x=e, y=avg, color='rec4'), data=rec4) +
  geom_line(aes(x=e, y=avg, color='rec8'), data=rec8) +
  geom_line(aes(x=e, y=avg, color='norm4'), data=norm4) +
  geom_line(aes(x=e, y=avg, color='norm8'), data=norm8) +
  labs(x='Episodes', y='Avearge Score') +
  ylim(100, 400)

##########################################################################
########################### Best vs Original #############################
ggplot(cl1, aes(x=e)) + 
  geom_line(aes(y=s, color="Score"), group=1) +
  geom_line(aes(y=avg, color="Avg Score"), group=1) + 
  labs(color="Legend text")

ggplot(norm8, aes(x=e)) + 
  geom_line(aes(y=s, color="Score"), group=1) +
  geom_line(aes(y=avg, color="Avg Score"), group=1) + 
  labs(color="Legend text")




