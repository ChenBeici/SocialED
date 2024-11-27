## 2024 11/10 徐智霖 

记录之前的模型可以顺利跑通四个数据集(按照_init_的顺序)

## CLKD 
在preprocesser里面对数据集进行处理 (我理解这里是调用了initial_lang, graph_lang等参数，但是我修改之后确实还是报下面的错)
本身支持event2018 French
Arabic_twitter arabic
Event2012  English
但是French运行的时候会显示报错
![1731221805309](image/SocialED/1731221805309.png)
arabic运行的时候会显示报错
![1731221829291](image/SocialED/1731221829291.png)

## KPGNN (四个数据集都跑通)
重点在于数据集必须得有created_at,tweet_id,user_mentions, user_id, sampled_words列

可以顺利跑通event2012，event2018, arabic_twitter

maven数据集因为没有这五列，目前的处理如下，增加了之后虽然可以跑通，但是我感觉这样的处理很粗糙
![1731222989708](image/SocialED/1731222989708.png)

## FINEVENT
![1731223197521](image/SocialED/1731223197521.png)
主要使用了：filtered_words，created_at，tweet_id, user_mentions, user_id, entities, sampled_words, event_id
初步运行可以成功执行但是执行时间太长并没有彻底执行完毕，但是按理来说我们的数据集都包括这些列应该是一个能跑通其他都能跑通

## QSGNN
主要用到的列有: filtered_words, created_at, tweet_id, user_mentions, user_id, entities, hashtags

maven数据集没有hashtags，看event2018数据集hashtags大部分为空
所以在dataloader中我也直接添加一列并且全部置为空
目前跑通了event2012

## HCRC

用到的数据列有user_loc列，但是只有event2012有这个列，所以还需要对其他三个原数据集继续修


## UCLSED
用到的列有：tweet_id, user_mentions, text, hashtags, entities, urls, filtered_words, created_at, event_id

所以maven数据集还得添加urls列

## RPLMSED
在经过前面的处理之后初步运行没有报错

## HISEVENT
尚未处理




上述提到的修改在dataloader函数中初步实现了