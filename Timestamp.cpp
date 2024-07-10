#include "Timestamp.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

// Function to get current time as a formatted string
std::string Timestamp::getCurrentTimeStamp() {
    // Get current time as a time_point
    auto now = std::chrono::system_clock::now();

    // Convert to time_t for formatting
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    // Convert to tm structure for formatting using localtime_s
    std::tm now_tm;
    localtime_s(&now_tm, &now_time_t);

    // Create a string stream to format the time
    std::stringstream ss;
    ss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");

    return ss.str();
}
