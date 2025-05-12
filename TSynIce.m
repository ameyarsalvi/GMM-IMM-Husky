%Convert TSync data --> to CSV

% Loop through each entry in TSyn (assumed to be from 1 to 11)
for i = 1:length(TSyn)
    % Get the current struct
    dataStruct = TSyn(i).data;

    % Get field names (e.g., 'time', 'pos_x', etc.)
    fieldNames = fieldnames(dataStruct);

    % Determine the number of samples
    n = length(dataStruct.(fieldNames{1}));  % assumes all fields have same or compatible length

    % Initialize table
    T = table();

    % Add each field to the table
    for j = 1:length(fieldNames)
        fieldData = dataStruct.(fieldNames{j});
        % Ensure it's a column vector
        if isrow(fieldData)
            fieldData = fieldData';
        end
        T.(fieldNames{j}) = fieldData;
    end

    % Define filename
    filename = sprintf('TSyn_%02d.csv', i);

    % Write to CSV
    writetable(T, filename);
    fprintf('Saved %s\n', filename);
end
