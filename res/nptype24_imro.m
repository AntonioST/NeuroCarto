function nptype24_imro

% Build imro tables for some useful four-shank combinations;
% also plot that selection.
%
% Output is saved in the directory where this script is run.
%
% patternType = 0 all sites on "shankChoice" starting from "botRow", 0-448
% patternType = 1 horizontal stripe of 96-channel height across all four
%                   shanks starting from "botRow", valid values = 0-592
% patternType = 2 two horizontal stripes of 48-channels across all 4 shanks;
%                   one starting from botRow, the other as close as possible
%                   to botRow_secondary
% patternType = 3 freeform assignment of shank and bank for 8 channel groups


patternType = 0;
shankChoice = 0;          % 0-3, needed for patternType 0
botRow =  0;              % used for single stripe (patternType 1), primary stripe in 2-stripe (patternType 2)
botRow_secondary = 320;   % used for 2nd stripe in two stripe (patternType 2)
refElec = 0;              % 0 for external, 1-4 for tip reference on shank 0-3


% for patternType = 3, edit arrays of shank + bank in case 3 below
blockShank = [0,1,3,0,1,2,3,0];
blockBank = [0,0,0,0,0,0,0,0];


shank = zeros(384,1,'single');
bank = zeros(384,1,'single');
elecInd = zeros(384,1,'single');
chans = zeros(384,1,'int32');

chans(:,1) = 0:383; % 384 channels
bMapOK = 1;

switch patternType

    case 0
        % make a map with all sites on one shank, starting from electrode row n
        nameStr = sprintf( 'NPtype24_shank%d_botRow%d_ref%d', shankChoice, botRow, refElec );
        shank = shank + shankChoice;
        elecInd = botRow*2:(botRow*2 + 383);
        for i = 1:numel(elecInd)
            [bank(i), chans(i)] = ElecToChan( shank(i), elecInd(i) );
        end

    case 1
        % horizontal stripe of 2 channel blocks (96 sites) across all four shanks
        shElecInd = (botRow*2:(botRow*2 + 95)); % these are the electrode indices on each shank
        nameStr = sprintf( 'NPtype24_hStripe_botRow%d_ref%d', botRow, refElec );
        % loop over shanks; for each, calculate the channels that correspond
        % to these electrode indices
        nE = 96; % electrodes in pattern per shank
        for sI = 0:3
            for i = 1:nE
                gEInd = sI*nE + i; % current electrode index for whole probe, plus one for MATLAB
                elecInd(gEInd) = shElecInd(i); % electrode index in whole selected set
                shank(gEInd) = sI;
                [bank(gEInd), chans(gEInd)] = ElecToChan( sI, elecInd(gEInd) );
                % fprintf("%d,%d,%d\n", bank(gEInd), chans(gEInd), elecInd(gEInd) );
            end
        end

    case 2
        % two horizontal stripes with 48 channels, 24 rows each, across all 4 shanks
        % botRow will be used as given; secondary_botRow will be shifted to
        % the nearest legal row -- this shift can be as large as 24 rows
        % nearest legal row for 2nd stripe is botRow + (2*n + 1)*24, where n is an integer
        n = round(((((botRow_secondary - botRow)/24) - 1)/2),0);
        bot2_legal = botRow + (2*n + 1)*24;

        shElecInd = [botRow*2:(botRow*2 + 47), bot2_legal*2:(bot2_legal*2 + 47)]; % these are the electrode indices on each shank
        shElecInd = sort(shElecInd);
        nameStr = sprintf( 'NPtype24_s1_%d_s2_%d_ref%d', botRow, bot2_legal, refElec );
        % loop over shanks; for each, calculate the channels that correspond
        % to these electrode indices
        nE = 96; % electrodes in pattern per shank
        for sI = 0:3
            for i = 1:nE
                gEInd = sI*nE + i; % current electrode index for whole probe, plus one for MATLAB
                elecInd(gEInd) = shElecInd(i); % electrode index in whole selected set
                shank(gEInd) = sI;
                [bank(gEInd), chans(gEInd)] = ElecToChan( sI, elecInd(gEInd) );
                % fprintf("%d,%d,%d\n", bank(gEInd), chans(gEInd), elecInd(gEInd) );
            end
        end

    case 3
        % specify shank and bank for each of the 8 allowed blocks.
        % note that these blocks appear in different parts of the bank on each shank
        % check figure 2 in the manual for where those blocks will be located
        nameStr = sprintf( 'NPtype24_byGroup_ref%d', refElec );
        for bI = 0:7
            for chanInBlock = 0:47  % 48 channels in each block
                gEInd = bI*48 + chanInBlock + 1; % current electrode index for whole probe, plus one for MATLAB
                shank(gEInd) = blockShank(bI+1);
                bank(gEInd) = blockBank(bI+1);
                chans(gEInd) = bI*48 + chanInBlock;
                elecInd(gEInd) = ChanToElec( shank(gEInd), bank(gEInd), bI, chanInBlock );
            end
        end


    otherwise
        fprintf('unknown pattern type\n');
        return;
end

% warn if there are duplicate channels
for i = 1:384
    if sum( chans(1:i-1)==chans(i) ) > 0
        fprintf( "duplicate channels => impossible map\n" );
        bMapOK = 0;
    end
end

if bMapOK
    % open a new file wherever we are
    fileName = [nameStr,'.imro'];
    nmID = fopen(fileName,'w');

    [chans,sortI] = sort(chans);
    bank = int32(bank(sortI));
    shank = int32(shank(sortI));
    elecInd = elecInd(sortI);

    % imro table
    % print first entry, specifying probe type and number of channels
    fprintf(nmID,'(%d,%d)', 24, 384);
    for i = 1:numel(chans)
        fprintf(nmID,'(%d %d %d %d %d)', chans(i), shank(i), bank(i), refElec, elecInd(i) );
    end
    fprintf(nmID, '\n');

    fclose(nmID);

    % make a plot of all the electrode positions
    [~,~,~] = PlotElec24(shank, bank, chans, elecInd);
end

end

function [ chans, chanPos, chanShank ] = PlotElec24( shank, bank, chans, elecInd )

    % NP 2.0 MS (4-shank), probe type 24 electrode positions
    nElec = 1280;   % per shank; pattern repeats for the four shanks
    vSep = 15;      % in um
    hSep = 32;

    elecPos = zeros(nElec, 2);

    elecPos(1:2:end,1) = 0;                % sites 0,2,4...
    elecPos(2:2:end,1) =  hSep;            % sites 1,3,5...

    % fill in y values
    viHalf = (0:(nElec/2-1))';                % row numbers
    elecPos(1:2:end,2) = viHalf * vSep;       % sites 0,2,4...
    elecPos(2:2:end,2) = elecPos(1:2:end,2);  % sites 1,3,5...

    chanPos = elecPos(elecInd+1,:);
    chanShank = shank;

    % make a plot of all the electrode positions
    figure(1)
    shankSep = 250;
    for sI = 0:3
        cc = find(shank == sI);
        scatter( shankSep*sI + elecPos(:,1), elecPos(:,2), 30, 'k', 'square' ); hold on;
        scatter( shankSep*sI + chanPos(cc,1), chanPos(cc,2), 20, 'b', 'square', 'filled' ); hold on;
    end
    xlim([-16,3*shankSep+64]);
    ylim([-10,10000]);
    title('NP2.0 MS shank view');
    hold off;

end

function [bank, chan] = ElecToChan( shank, elecInd )

% electrode to channel map
elecMap = zeros(4,8,'single');
elecMap(1,:) = [0,2,4,6,5,7,1,3];
elecMap(2,:) = [1,3,5,7,4,6,0,2];
elecMap(3,:) = [4,6,0,2,1,3,5,7];
elecMap(4,:) = [5,7,1,3,0,2,4,6];

bank = floor(elecInd/384);

% which block within the bank?
blockID = floor((elecInd - bank*384)/48);

% which channel within the block?
subBlockInd = mod((elecInd - bank*384), 48);

% get the channel number for that shank, bank, and block combo
chan = 48*elecMap(shank+1, blockID+1) + subBlockInd;

end

function elecInd = ChanToElec( shank, bank, block, chanInBlock )

% blockMap blockIndex -> position on shank
blockMap = zeros(4,8,'single');
blockMap(1,:) = [0,6,1,7,2,4,3,5];
blockMap(2,:) = [6,0,7,1,4,2,5,3];
blockMap(3,:) = [2,4,3,5,0,6,1,7];
blockMap(4,:) = [4,2,5,3,6,0,7,1];

% bank, shank, block, chanInBlock and electInd all zero based
elecInd = bank*384 + 48*(blockMap(shank+1,block+1)) + chanInBlock;

end


