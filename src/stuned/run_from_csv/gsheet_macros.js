function FillRunCmd() {
    var runcmd = '<environment activation> && python <path_to_run_from_csv> ' +
    '--conda_env <env to use> --csv_path <URL>::<WORKSHEET>';
    FillCmd(runcmd, "cmd_to_run_exp")
};


function FillLabelCmd() {
    var runcmd = '<environment activation> && python <path_to_label_gsheet> ' +
    '--gsheet_url <URL>::<WORKSHEET>';
    FillCmd(runcmd, "cmd_to_prepare_for_labeling")
};


function FillCmd(runcmd, column_name) {
    var spreadsheet = SpreadsheetApp.getActive();
    spreadsheet.getRange('A:A').activate();
    spreadsheet.getActiveSheet().insertColumnsBefore(
        spreadsheet.getActiveRange().getColumn(), 1
    );
    spreadsheet.getActiveRange().offset(
        0,
        0,
        spreadsheet.getActiveRange().getNumRows(),
        1
    ).activate();
    spreadsheet.getRange('A1').activate();
    spreadsheet.getCurrentCell().setValue(column_name);
    spreadsheet.getRange('A2').activate();
    runcmd = runcmd.replace("<WORKSHEET>", GetSheetName().toString());
    runcmd = runcmd.replace("<URL>", GetSheetUrl().toString());
    spreadsheet.getCurrentCell().setValue(runcmd);
};


function GetSheetName() {
    return SpreadsheetApp.getActiveSpreadsheet().getActiveSheet().getName();
}


function GetSheetUrl() {
    return SpreadsheetApp.getActiveSpreadsheet().getUrl();
}


function MakeAllRunnable() {
    var range = SpreadsheetApp.getActiveSpreadsheet().getActiveRange()
    ReplaceInColumn(range, "0", "1")
}


function MakeAllNonRunnable() {
    var range = SpreadsheetApp.getActiveSpreadsheet().getActiveRange()
    ReplaceInColumn(range, "1", "0")
}


function MakeAllNonesDefault() {
    var range = SpreadsheetApp.getActiveSpreadsheet().getActiveRange()
    ReplaceInColumn(range, "", "DEFAULT")
}


function ReplaceInColumn(range, old_val, new_val) {
    var v = range.getValues();
    for ( var r = 0; r < v.length; ++r) {
        cur_val_as_str = v[r][0].toString()
        if (cur_val_as_str == old_val) {
            v[r][0] = new_val;
        }
    }
    range.setValues(v);
}


/** Expand rows */


function expandRows() {
    var listData = parseSelectedRows();
    listData = listData.map(list => convertToNestedLists(list));
    listData = listData.map(list => cartesianProduct(list));
    listData = flattenListOfLists(listData);
    insertListAsRows(listData);
}


function parseSelectedRows() {
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = spreadsheet.getActiveSheet();

    // Get the selected range
    var selectedRange = sheet.getActiveRange();

    // Get the values of the selected range as a two-dimensional array
    var values = selectedRange.getValues();

    // Convert the two-dimensional array into a list of lists of strings
    var listOfLists = values.map(function(row) {
        return row.map(function(cell) {
            return String(cell); // Convert cell value to string
        });
    });

    // Log the resulting list of lists in the Apps Script editor's log
    Logger.log(listOfLists);
    return listOfLists;
  }


function insertListAsRows(listData) {
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = spreadsheet.getActiveSheet();

    // Get the selected range
    var selectedRange = sheet.getActiveRange();

    // Insert the list of lists as rows below the selected range
    sheet.insertRowsAfter(selectedRange.getLastRow(), listData.length);

    // Get the range of the newly inserted rows
    var newRange = sheet.getRange(
        selectedRange.getLastRow() + 1,
        1,
        listData.length,
        listData[0].length
    );

    // Set the values of the newly inserted rows
    newRange.setValues(listData);
  }


function convertToNestedLists(strings) {
    // Use the map function to wrap each string in a list of length 1
    const listOfLists = strings.map(str => parseAnyStringToList(str));
    return listOfLists;
  }


function cartesianProduct(lists) {
    function cartesianRecursive(arr, i) {
        if (i === lists.length) {
            results.push(arr.slice()); // Clone the current combination
            return;
        }

        for (const value of lists[i]) {
            arr.push(value);
            cartesianRecursive(arr, i + 1);
            arr.pop(); // Backtrack to try the next value
        }
    }

    const results = [];
    cartesianRecursive([], 0);
    return results;
}


function flattenListOfLists(lists) {
    return lists.reduce(function (result, list) {
        return result.concat(list);
    }, []);
}


function parseStringToList(inputString) {
    // Use regular expression to match values inside curly braces and split them by '|'
    const matches = inputString.match(/\{([^}]+)\}/);

    if (matches && matches[1]) {
        const values = matches[1].trim().split(/\s*\|\s*/);
        return values;
    }

    // Return an empty array if no match is found
    return [];
  }


function parseAnyStringToList(inputString) {
    if (inputString.startsWith("{") && inputString.endsWith("}")) {
        if (inputString.includes("<") && inputString.includes(">")) {
            return parseRange(inputString);
        }
        else {
            return parseStringToList(inputString);
        }
    } else {
        return [inputString];
    }
}


function parseRange(inputString) {
    // Regular expression to match the range pattern
    const pattern = /\{<(\d+)\s+(\d+)\s+(\d+)(?: (True|False))?>\}/;

    // Match the pattern in the inputString
    const match = inputString.match(pattern);

    // Check if the inputString matches the pattern
    if (match) {
        const start = parseInt(match[1]);
        const end = parseInt(match[2]);
        const step = parseInt(match[3]);

        var isLogarithmic = false;
        if (match[4] == "True" || match[4] == "true") {
            isLogarithmic = true;
        }

    if (start <= end && step > 0) {
        const result = [];
        if (isLogarithmic) {
            for (let i = start; i <= end; i *= step) {
                result.push(i);
            }
        } else {
            for (let i = start; i <= end; i += step) {
                result.push(i);
            }
        }
        return result;
    } else {
        console.error("Invalid range parameters.");
        return null;
    }
    } else {
        console.error("Invalid input format.");
        return null;
    }
}
