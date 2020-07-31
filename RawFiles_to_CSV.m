j = 1;
while j <= 20
    x = 'eeg_record';
    jtoString = num2str(j);
    matEnding = '.mat';
    cat2 = [x jtoString matEnding];
    data = load(cat2);
    data = data.o.data;
    j = j + 1; %code to save 10 minutes of data into .csv files of 10 sec readings
    i = 1;
    while i <= 76800
        tempMatrix = data(i:i+1280,:);
        name = 'focused_';
        inBetween = '_';
        num = num2str(round(i/1280) + 1);
        ending = '.csv';
        cat = [name jtoString inBetween num ending];
        disp(cat);
        csvwrite(cat, tempMatrix)
        i = i + 1280;
    end
    
    while i <= 153600   % readings from 76800 < x <= 153600
        tempMatrix = data(i:i+1280,:);
        name = 'defocused_';
        inBetween = '_';
        num = num2str(round(i/1280) - 59);
        ending = '.csv';
        cat = [name jtoString inBetween num ending];
        disp(cat);
        csvwrite(cat, tempMatrix)
        i = i + 1280;
    end
    
    while i <= 230400   % readings from 153600 < x <= 230400
        tempMatrix = data(i:i+1280,:);
        name = 'drowsy_';
        inBetween = '_';
        num = num2str(round(i/1280) - 119);
        ending = '.csv';
        cat = [name toString inBetween num ending];
        disp(cat);
        csvwrite(cat, tempMatrix)
        i = i + 1280;
    end
end
