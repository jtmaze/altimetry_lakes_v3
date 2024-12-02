library(tidyverse)
library(wesanderson)

df <- read.csv('./data/IS2_obscnts.csv')

df$roi_name <- ifelse(df$roi_name == "YKflats", "Yukon Flats",
                      ifelse(df$roi_name == "YKdelta", "Yukon Delta",
                             ifelse(df$roi_name == "MRD_TUK_Anderson", "Mckenzie River Delta / Tuktoyaktuk Peninsula",
                                    ifelse(df$roi_name == "AKCP", "Alaska Coastal Plain", df$roi_name)))) 

df <- df %>%
  rename(
    `1. One summer observation all years (2019-2023)` = matched_percent,
    `2. At least one summer observation per year` = obs5_percent,
    `3. Multiple summer observations per year` = obs10_percent
  )


df_long <- df %>%
  pivot_longer(cols = c(`1. One summer observation all years (2019-2023)`,
                        `2. At least one summer observation per year`,
                        `3. Multiple summer observations per year`),
               names_to = "Metric", values_to = "Percentage") %>% 
  mutate(roi_name_total = paste0(roi_name, " total PLD lakes = ", total_lakes))



# Updated ggplot code
ggplot(df_long, 
       aes(x = roi_name_total, y = Percentage, fill = roi_name_total)) +
  geom_bar(stat = "identity", width=0.5) +
  facet_wrap(~ Metric, nrow = 3, scales = 'free_y') +
  scale_fill_manual(values = wes_palette("Chevalier1", 
                        n = length(unique(df_long$roi_name)), 
                        type = "continuous")) +
  labs(title = "Percentage of PLD lake with ICESat-2 summer WSE observations",
       x = NULL,
       y = "% of PLD lakes", 
       fill = NULL) +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),    
    axis.title.x = element_blank(),   
    legend.position = "bottom",
    plot.title = element_text(face = "bold",    
                              hjust = 0.5,      
                              size = 14),       
    strip.text = element_text(face = "bold")
  )
