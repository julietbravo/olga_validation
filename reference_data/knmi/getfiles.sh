# Download the 2011-2020 data from KNMI

stations=(210 215 225 235 240 242 249 251 257 260 265 267 269 270 273 275 277 278 279 280 283 286 290 310 319 323 330 340 344 348 350 356 370 375 377 380 391)

# Download files
for i in ${stations[@]}; do
    wget http://www.knmi.nl/klimatologie/uurgegevens/datafiles/${i}/uurgeg_${i}_2011-2020.zip
done

# Unzip
for i in ${stations[@]}; do
    unzip uurgeg_${i}_2011-2020.zip # Macbook
done

