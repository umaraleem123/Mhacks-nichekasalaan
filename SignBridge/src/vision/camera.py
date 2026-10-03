"""Open a webcam for the capture loop.

Prefers a Logitech Brio (the camera used with this project) and otherwise
opens the default camera. On Windows the default Media Foundation backend can
sit for a long time before returning frames from the Brio, so capture goes
through OpenCV's DirectShow backend there. Other platforms use OpenCV's
default backend.

Pass an explicit index to skip the preference order and open that device.
"""

from __future__ import annotations

import sys

import cv2

REQUESTED_WIDTH = 1280
REQUESTED_HEIGHT = 720

# More specific names first, so "Brio 101" wins over a generic Logitech device.
_PREFERRED_NAME_MARKERS = ("brio 101", "brio", "logitech")
_MAX_PROBED_INDEX = 4

CAMERA_OPEN_ERROR = """Error: could not open a webcam.

The app tries a Logitech Brio first, then the default camera.

Things to check:
  1. Camera permission. macOS: System Settings > Privacy & Security > Camera,
     and enable the terminal app you launched this from. Windows: Settings >
     Privacy & security > Camera, and allow desktop apps to access the camera.
  2. Another application may be holding the camera. Quit video calls, browser
     tabs, and recording tools, then try again.
  3. On a desktop machine, confirm an external webcam is plugged in."""

CAMERA_LOST_ERROR = """Error: lost the connection to the webcam.

The camera stopped returning frames. It may have been unplugged or claimed by
another application. Reconnect it and run the app again."""


def open_camera(index: int | None = None) -> tuple[cv2.VideoCapture, str] | None:
    """Open a camera and return it with a display name.

    With no index, opens the Brio when it is connected and otherwise the
    default camera. With an index, opens exactly that device. Returns None
    after printing why the open failed.
    """
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

    if index is not None:
        camera = _open_index(index, backend)
        if camera is None:
            print(CAMERA_OPEN_ERROR, file=sys.stderr)
            return None
        name = _index_label(index)
        print(f"Using camera: {name}")
        return camera, name

    names = _capture_device_names()
    if names:
        candidates = _candidate_order(names)
    else:
        candidates = [
            (probed, _index_label(probed)) for probed in range(_MAX_PROBED_INDEX + 1)
        ]

    for candidate_index, name in candidates:
        camera = _open_index(candidate_index, backend)
        if camera is not None:
            print(f"Using camera: {name}")
            return camera, name

    print(CAMERA_OPEN_ERROR, file=sys.stderr)
    return None


def _index_label(index: int) -> str:
    return "default camera" if index == 0 else f"camera {index}"


def _candidate_order(names: list[str]) -> list[tuple[int, str]]:
    """Brio / Logitech devices first, then every other device, default first."""
    preferred: list[tuple[int, int, str]] = []
    others: list[tuple[int, str]] = []
    for index, name in enumerate(names):
        rank = _preference_rank(name)
        if rank is None:
            others.append((index, name))
        else:
            preferred.append((rank, index, name))
    preferred.sort()
    return [(index, name) for _, index, name in preferred] + others


def _preference_rank(name: str) -> int | None:
    lowered = name.lower()
    for rank, marker in enumerate(_PREFERRED_NAME_MARKERS):
        if marker in lowered:
            return rank
    return None


def _open_index(index: int, backend: int) -> cv2.VideoCapture | None:
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        return None

    # Requests only; the driver picks the nearest supported mode.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
    return camera


def _capture_device_names() -> list[str] | None:
    """Capture-device names in OpenCV index order, or None if unknown.

    Windows DirectShow order matches ``cv2.CAP_DSHOW`` indexes. Other
    platforms have no portable name listing, so the caller probes indexes.
    """
    if sys.platform != "win32":
        return None
    try:
        return _windows_directshow_names()
    except OSError:
        return None


def _windows_directshow_names() -> list[str]:
    """Friendly names of DirectShow video-input devices, in device-index order."""
    import ctypes
    from ctypes import POINTER, byref, c_void_p, c_wchar_p
    from ctypes.wintypes import BYTE, DWORD, ULONG, WORD

    ole32 = ctypes.WinDLL("ole32")
    oleaut32 = ctypes.WinDLL("oleaut32")
    hresult = ctypes.c_long
    clsctx_inproc_server = 1
    coinit_apartment = 2
    rpc_e_changed_mode = -2147417850
    vt_bstr = 8

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", DWORD),
            ("Data2", WORD),
            ("Data3", WORD),
            ("Data4", BYTE * 8),
        ]

    class VARIANT(ctypes.Structure):
        class _Value(ctypes.Union):
            _fields_ = [("bstrVal", c_void_p), ("llVal", ctypes.c_longlong)]

        _fields_ = [
            ("vt", WORD),
            ("wReserved1", WORD),
            ("wReserved2", WORD),
            ("wReserved3", WORD),
            ("value", _Value),
            ("_pad", ctypes.c_ulonglong),
        ]

    ole32.CLSIDFromString.argtypes = [c_wchar_p, POINTER(GUID)]
    ole32.CLSIDFromString.restype = hresult
    ole32.CoInitializeEx.argtypes = [c_void_p, DWORD]
    ole32.CoInitializeEx.restype = hresult
    ole32.CoUninitialize.argtypes = []
    ole32.CoCreateInstance.argtypes = [
        POINTER(GUID),
        c_void_p,
        DWORD,
        POINTER(GUID),
        POINTER(c_void_p),
    ]
    ole32.CoCreateInstance.restype = hresult
    oleaut32.VariantClear.argtypes = [POINTER(VARIANT)]
    oleaut32.VariantClear.restype = hresult

    create_enum = ctypes.WINFUNCTYPE(hresult, c_void_p, POINTER(GUID), POINTER(c_void_p), DWORD)
    enum_next = ctypes.WINFUNCTYPE(hresult, c_void_p, ULONG, POINTER(c_void_p), POINTER(ULONG))
    release = ctypes.WINFUNCTYPE(ULONG, c_void_p)
    bind_to_storage = ctypes.WINFUNCTYPE(
        hresult, c_void_p, c_void_p, c_void_p, POINTER(GUID), POINTER(c_void_p)
    )
    prop_read = ctypes.WINFUNCTYPE(hresult, c_void_p, c_wchar_p, POINTER(VARIANT), c_void_p)

    def make_guid(text: str) -> GUID:
        value = GUID()
        if ole32.CLSIDFromString(text, byref(value)) != 0:
            raise OSError(f"invalid GUID {text}")
        return value

    def vtable_method(ptr: c_void_p, index: int, prototype):
        table = ctypes.cast(ptr, POINTER(POINTER(c_void_p))).contents
        return ctypes.cast(table[index], prototype)

    def release_com(ptr: c_void_p) -> None:
        vtable_method(ptr, 2, release)(ptr)

    def friendly_name(moniker: c_void_p) -> str:
        bag = c_void_p()
        # IMoniker::BindToStorage is vtable slot 9.
        hr = vtable_method(moniker, 9, bind_to_storage)(
            moniker, None, None, byref(property_bag), byref(bag)
        )
        if hr != 0 or not bag:
            raise OSError(f"BindToStorage failed: {hr:#x}")
        variant = VARIANT()
        try:
            hr = vtable_method(bag, 3, prop_read)(bag, "FriendlyName", byref(variant), None)
            if hr != 0 or variant.vt != vt_bstr or not variant.value.bstrVal:
                raise OSError(f"FriendlyName read failed: {hr:#x}")
            return ctypes.wstring_at(variant.value.bstrVal)
        finally:
            if variant.vt == vt_bstr:
                oleaut32.VariantClear(byref(variant))
            release_com(bag)

    # S_FALSE (1) means this thread already initialized COM.
    init_hr = ole32.CoInitializeEx(None, coinit_apartment)
    owns_com = init_hr in (0, 1)
    if not owns_com and init_hr != rpc_e_changed_mode:
        raise OSError(f"CoInitializeEx failed: {init_hr:#x}")

    try:
        dev_enum = c_void_p()
        hr = ole32.CoCreateInstance(
            byref(make_guid("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")),
            None,
            clsctx_inproc_server,
            byref(make_guid("{29840822-5B84-11D0-BD3B-00A0C911CE86}")),
            byref(dev_enum),
        )
        if hr != 0:
            raise OSError(f"CoCreateInstance failed: {hr:#x}")

        enum_moniker = c_void_p()
        category = make_guid("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")
        hr = vtable_method(dev_enum, 3, create_enum)(
            dev_enum, byref(category), byref(enum_moniker), 0
        )
        release_com(dev_enum)
        if hr != 0 or not enum_moniker:
            return []

        property_bag = make_guid("{55272A00-42CB-11CE-8135-00AA004BB851}")
        names: list[str] = []
        try:
            while True:
                moniker = c_void_p()
                fetched = ULONG(0)
                hr = vtable_method(enum_moniker, 3, enum_next)(
                    enum_moniker, 1, byref(moniker), byref(fetched)
                )
                if hr != 0 or not fetched.value or not moniker:
                    break
                names.append(friendly_name(moniker))
                release_com(moniker)
        finally:
            release_com(enum_moniker)
        return names
    finally:
        if owns_com:
            ole32.CoUninitialize()
