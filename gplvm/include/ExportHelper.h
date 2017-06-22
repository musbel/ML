#ifndef EXPORTHELPER_H
#define EXPORTHELPER_H

#ifdef _WIN32
#  define ML_API __declspec( dllexport )
#else
#  define ML_API
#endif

#endif
